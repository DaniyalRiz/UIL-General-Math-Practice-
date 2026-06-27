// Admin-only: Step 2 of the PDF import pipeline. Takes a batch_id whose
// import_batches row already has status='transcribed' (written by
// import-test-pdf / step 1) and reads `boundaries_json` back out of the
// database to solve each question and write draft_questions rows for human
// review. Never touches `questions` and never sets review_status to
// anything but 'pending'.
//
// Text-only questions carry no PDF attachment, so they're cheap and are all
// solved together in the background via EdgeRuntime.waitUntil, kicked off
// exactly once per batch (on the first call, detected via the batch's status
// still being 'transcribed'). needs_image questions re-attach a PDF page and
// are expensive enough to risk the account's 30,000-input-tokens/minute
// limit, so this function solves at most one of them per invocation and lets
// the browser drive it forward with a paced delay between calls -- the same
// pattern as step 1's per-page transcription. "What's left to do" is derived
// by diffing boundaries_json against the draft_questions rows already
// inserted, so this needs no separate progress column and is safe to call
// again if a previous response was lost.
import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { createClient } from "jsr:@supabase/supabase-js@2";
import { encodeBase64 } from "jsr:@std/encoding/base64";
import { PDFDocument } from "npm:pdf-lib@1.17.1";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const ANTHROPIC_API_KEY = Deno.env.get("ANTHROPIC_API_KEY");
const ANTHROPIC_MODEL = "claude-sonnet-4-6";
const CALL_TIMEOUT_MS = 100_000;

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, content-type, apikey, x-client-info",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...CORS_HEADERS },
  });
}

const SOLVE_TOOL = {
  name: "solve_question",
  description: "Solve this single math competition question and give a brief worked solution.",
  strict: true,
  input_schema: {
    type: "object",
    properties: {
      extracted_answer: { type: "string", description: "The correct choice, exactly matching one entry in choices" },
      explanation: {
        type: "string",
        maxLength: 500,
        description:
          "Concise worked solution, at most 2-3 sentences -- the key steps and final answer only, not a full essay. " +
          "Use \\( ... \\) for inline math and \\[ ... \\] for display math (never $ or $$) -- the renderer only recognizes backslash delimiters.",
      },
    },
    required: ["extracted_answer", "explanation"],
    additionalProperties: false,
  },
};

type QuestionBoundary = {
  original_question_number: number;
  title: string;
  topic: string;
  difficulty: string;
  question: string;
  choices: string[];
  tags: string[];
  needs_image: boolean;
  image_alt: string;
  _pageIndex: number;
};

function parseAnswerKey(raw: string): Map<number, string> {
  const map = new Map<number, string>();
  // Accepts "1. C" / "1) C" / "1:C" / bare comma-separated list "C, C, D, ..."
  const numbered = [...raw.matchAll(/(\d+)[).:]\s*([A-E])\b/gi)];
  if (numbered.length > 0) {
    for (const m of numbered) map.set(parseInt(m[1], 10), m[2].toUpperCase());
    return map;
  }
  const bare = raw.match(/[A-E]/gi);
  if (bare) {
    bare.forEach((letter, idx) => map.set(idx + 1, letter.toUpperCase()));
  }
  return map;
}

function choiceLetter(choiceText: string): string | null {
  const m = choiceText.match(/^\(?([A-E])\)?/i);
  return m ? m[1].toUpperCase() : null;
}

async function streamToolCall(
  body: Record<string, unknown>,
  signal: AbortSignal,
  label: string,
): Promise<{ stopReason: string; toolInput: Record<string, unknown> }> {
  const resp = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    signal,
    headers: {
      "content-type": "application/json",
      "x-api-key": ANTHROPIC_API_KEY!,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const errText = await resp.text();
    throw new Error(`Claude API error ${resp.status}: ${errText}`);
  }
  if (!resp.body) throw new Error("Claude API returned no response body for the streamed request");

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let partialJson = "";
  let stopReason = "end_turn";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const dataStr = line.slice(6).trim();
      if (!dataStr) continue;
      let evt: { type?: string; delta?: { type?: string; partial_json?: string; stop_reason?: string } };
      try {
        evt = JSON.parse(dataStr);
      } catch {
        continue;
      }
      if (evt.type === "content_block_delta" && evt.delta?.type === "input_json_delta") {
        partialJson += evt.delta.partial_json ?? "";
      } else if (evt.type === "message_delta" && evt.delta?.stop_reason) {
        stopReason = evt.delta.stop_reason;
      }
    }
  }

  if (!partialJson) throw new Error(`[${label}] Claude did not stream any tool input. stop_reason=${stopReason}`);

  let toolInput: Record<string, unknown>;
  try {
    toolInput = JSON.parse(partialJson);
  } catch (e) {
    throw new Error(`Failed to parse streamed tool input as JSON: ${e instanceof Error ? e.message : String(e)}`);
  }

  return { stopReason, toolInput };
}

function withTimeout<T>(fn: (signal: AbortSignal) => Promise<T>): Promise<T> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), CALL_TIMEOUT_MS);
  return fn(controller.signal).finally(() => clearTimeout(timeoutId));
}

// Runs `fn` over every item like Promise.allSettled, but never more than
// `limit` calls in flight at once -- keeps the per-minute token/request burst
// against the Anthropic API small instead of firing everything at once. Only
// used for text-only questions, which carry no PDF attachment and so don't
// risk the rate limit the way needs_image questions do.
async function runWithConcurrency<T, R>(
  items: T[],
  limit: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<PromiseSettledResult<R>[]> {
  const results: PromiseSettledResult<R>[] = new Array(items.length);
  let nextIndex = 0;
  async function worker() {
    while (true) {
      const i = nextIndex++;
      if (i >= items.length) return;
      try {
        results[i] = { status: "fulfilled", value: await fn(items[i], i) };
      } catch (reason) {
        results[i] = { status: "rejected", reason };
      }
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, () => worker()));
  return results;
}

type SupabaseClient = ReturnType<typeof createClient>;

async function markFailed(db: SupabaseClient, batchId: string, message: string) {
  await db
    .from("import_batches")
    .update({ status: "failed", error_message: message, finished_at: new Date().toISOString(), updated_at: new Date().toISOString() })
    .eq("id", batchId);
}

// Re-derives the per-page PDFs from the audit copy step 1 uploaded to
// storage. Only needed for needs_image questions -- if it's unavailable for
// any reason, those questions are just solved text-only instead of hard-failing.
async function loadPagePdfs(db: SupabaseClient, sourcePdfPath: string | null): Promise<string[]> {
  if (!sourcePdfPath) return [];
  try {
    const { data, error } = await db.storage.from("test-pdfs").download(sourcePdfPath);
    if (error || !data) return [];
    const bytes = new Uint8Array(await data.arrayBuffer());
    const srcDoc = await PDFDocument.load(bytes);
    const pageCount = srcDoc.getPageCount();
    const pagePdfs: string[] = [];
    for (let i = 0; i < pageCount; i++) {
      const newDoc = await PDFDocument.create();
      const [copiedPage] = await newDoc.copyPages(srcDoc, [i]);
      newDoc.addPage(copiedPage);
      pagePdfs.push(encodeBase64(await newDoc.save()));
    }
    return pagePdfs;
  } catch {
    return [];
  }
}

async function solveQuestion(
  pagePdfs: string[],
  q: QuestionBoundary,
): Promise<{ extracted_answer: string; explanation: string }> {
  const content: unknown[] = [];
  // Only re-attach the page for questions that genuinely need the diagram --
  // keeps every other solve call cheap and fast.
  if (q.needs_image && pagePdfs[q._pageIndex]) {
    // Cached -- the same page gets re-sent byte-for-byte on every automatic
    // mismatch retry (and possibly for other questions sharing the page), so
    // without this every retry pays full input-token price for an identical
    // attachment. Cache hits cost roughly a tenth as much as a fresh send.
    content.push({
      type: "document",
      source: { type: "base64", media_type: "application/pdf", data: pagePdfs[q._pageIndex] },
      cache_control: { type: "ephemeral" },
    });
  }
  content.push({
    type: "text",
    text:
      `Solve this math competition question and show your real work.\n\n` +
      `Question ${q.original_question_number}: ${q.question}\n\n` +
      `Choices:\n${q.choices.join("\n")}\n\n` +
      (q.needs_image && pagePdfs[q._pageIndex] ? "The PDF page is attached above because this question depends on a diagram -- look at it carefully.\n" : "") +
      `Set extracted_answer to exactly one of the choices above, and explanation to a brief worked solution -- ` +
      `at most 2-3 sentences covering the key steps and final answer, not a full essay. ` +
      `Write any math using \\( ... \\) for inline and \\[ ... \\] for display -- never $ or $$, which the renderer will show as literal text.`,
  });

  const { stopReason, toolInput } = await withTimeout((signal) =>
    streamToolCall(
      {
        model: ANTHROPIC_MODEL,
        max_tokens: 4096,
        stream: true,
        tools: [SOLVE_TOOL],
        tool_choice: { type: "tool", name: "solve_question" },
        messages: [{ role: "user", content }],
      },
      signal,
      `solve#${q.original_question_number}`,
    )
  );

  if (stopReason === "refusal") throw new Error("Claude declined to solve this question (refusal)");
  const extracted_answer = toolInput.extracted_answer as string | undefined;
  const explanation = toolInput.explanation as string | undefined;
  if (!extracted_answer || !explanation) throw new Error("Claude did not return a complete solution");
  return { extracted_answer, explanation };
}

type SolvedFields = {
  extracted_answer: string;
  claude_solved_answer: string;
  answer: string;
  explanation: string;
  verification_status: "match" | "mismatch" | "unverified" | "no_answer_key";
  verification_notes: string | null;
};

async function solveOnce(
  q: QuestionBoundary,
  pagePdfs: string[],
  originalTest: string | null,
  answerKeyMap: Map<number, string>,
  hasAnswerKey: boolean,
  dedupKeys: Set<string>,
): Promise<SolvedFields> {
  let extracted_answer = "";
  let explanation = "[Automatic solving failed -- please solve manually.]";
  let solveError: string | null = null;
  try {
    const solved = await solveQuestion(pagePdfs, q);
    extracted_answer = solved.extracted_answer;
    explanation = solved.explanation;
  } catch (err) {
    solveError = err instanceof Error && err.name === "AbortError"
      ? `Claude did not finish solving this question within ${CALL_TIMEOUT_MS / 1000}s.`
      : err instanceof Error
        ? err.message
        : String(err);
  }

  const extractedLetter = choiceLetter(extracted_answer) ?? choiceLetter(q.choices[0] ?? "");
  const keyLetter = answerKeyMap.get(q.original_question_number);
  let verification_status: SolvedFields["verification_status"] = "unverified";
  let verification_notes: string | null = solveError ? `Solving failed: ${solveError}` : null;
  if (solveError) {
    verification_status = "unverified";
  } else if (!hasAnswerKey) {
    verification_status = "no_answer_key";
  } else if (!keyLetter) {
    verification_status = "unverified";
    verification_notes = "Answer key did not include this question number";
  } else if (extractedLetter && extractedLetter === keyLetter) {
    verification_status = "match";
  } else {
    verification_status = "mismatch";
    verification_notes = `Claude solved (${extractedLetter ?? "?"}), answer key says (${keyLetter})`;
  }

  if (originalTest && dedupKeys.has(`${originalTest}|${q.original_question_number}`)) {
    const dupeNote = "Possible duplicate: a question with this test/number already exists in the published bank.";
    verification_notes = verification_notes ? `${verification_notes} ${dupeNote}` : dupeNote;
  }

  const matchedChoice = q.choices.find((c) => choiceLetter(c) === (keyLetter ?? extractedLetter));

  return {
    extracted_answer,
    claude_solved_answer: extracted_answer,
    answer: matchedChoice ?? extracted_answer,
    explanation,
    verification_status,
    verification_notes,
  };
}

// Solves one question and computes its verification fields -- shared by the
// initial insert path (background text-only batch, single needs-image call)
// and the redo path (re-solving one already-inserted question in place).
// Retries automatically while the result is a genuine answer-key mismatch,
// so the admin never has to manually press Redo just to get Claude to try
// again -- capped so a question that's actually hard (not just unlucky)
// can't loop forever burning API calls. Only a mismatch is retried: a solve
// failure, a missing key, or an already-correct match has nothing to gain
// from hammering the API again.
const MAX_SOLVE_ATTEMPTS = 2;
async function solveAndBuildFields(
  q: QuestionBoundary,
  pagePdfs: string[],
  originalTest: string | null,
  answerKeyMap: Map<number, string>,
  hasAnswerKey: boolean,
  dedupKeys: Set<string>,
): Promise<SolvedFields> {
  let fields: SolvedFields = await solveOnce(q, pagePdfs, originalTest, answerKeyMap, hasAnswerKey, dedupKeys);
  let attempts = 1;
  while (fields.verification_status === "mismatch" && attempts < MAX_SOLVE_ATTEMPTS) {
    fields = await solveOnce(q, pagePdfs, originalTest, answerKeyMap, hasAnswerKey, dedupKeys);
    attempts++;
  }
  return fields;
}

// Solves one question end-to-end and inserts its draft_questions row
// immediately -- shared by both the background text-only batch and the
// single synchronous needs-image call. `dedupKeys` is the set of
// "original_test|original_question_number" pairs that already exist in the
// published `questions` table, so an admin re-importing the same test twice
// gets a clear warning instead of a silent duplicate.
async function solveAndInsertOne(
  db: SupabaseClient,
  batchId: string,
  q: QuestionBoundary,
  pagePdfs: string[],
  sourceLabel: string | null,
  originalTest: string | null,
  answerKeyMap: Map<number, string>,
  hasAnswerKey: boolean,
  dedupKeys: Set<string>,
): Promise<void> {
  const fields = await solveAndBuildFields(q, pagePdfs, originalTest, answerKeyMap, hasAnswerKey, dedupKeys);

  const row = {
    batch_id: batchId,
    title: q.title,
    topic: q.topic,
    difficulty: q.difficulty,
    source: sourceLabel ?? null,
    question: q.question,
    choices: q.choices,
    tags: q.tags ?? [],
    ...fields,
    needs_image: q.needs_image,
    image_alt: q.needs_image ? q.image_alt : null,
    original_test: originalTest ?? null,
    original_question_number: q.original_question_number,
    source_reference: sourceLabel ? `${sourceLabel} #${q.original_question_number}` : null,
    review_status: "pending",
  };

  // Insert immediately -- this is what makes results show up live in the
  // review UI instead of waiting for every question to finish.
  const { error: insertErr } = await db.from("draft_questions").insert(row);
  if (insertErr) throw new Error(`Failed to insert question ${q.original_question_number}: ${insertErr.message}`);
}

// Fired once via EdgeRuntime.waitUntil on the first call for a batch.
async function solveTextOnlyBatch(
  db: SupabaseClient,
  batchId: string,
  questions: QuestionBoundary[],
  sourceLabel: string | null,
  originalTest: string | null,
  answerKeyMap: Map<number, string>,
  hasAnswerKey: boolean,
  dedupKeys: Set<string>,
) {
  await runWithConcurrency(questions, 6, (q) =>
    solveAndInsertOne(db, batchId, q, [], sourceLabel, originalTest, answerKeyMap, hasAnswerKey, dedupKeys)
  );
  await finalizeIfDone(db, batchId);
}

// Checks whether every boundary now has a matching draft_questions row, and
// if so, flips the batch to its terminal status. Safe to call after every
// partial step -- a no-op until the last question lands.
async function finalizeIfDone(db: SupabaseClient, batchId: string) {
  const { data: batch } = await db.from("import_batches").select("boundaries_json").eq("id", batchId).single();
  const boundaries = (batch?.boundaries_json ?? []) as QuestionBoundary[];
  const { data: drafts } = await db
    .from("draft_questions")
    .select("verification_status, verification_notes")
    .eq("batch_id", batchId);
  if (!drafts || drafts.length < boundaries.length) return; // still in flight

  if (drafts.length === 0) {
    await markFailed(db, batchId, "All questions failed to solve -- check individual question errors and retry.");
    return;
  }

  const needsAttention = drafts.some(
    (r) =>
      r.verification_status === "mismatch" ||
      (r.verification_notes ?? "").startsWith("Solving failed:") ||
      (r.verification_notes ?? "").includes("Possible duplicate"),
  );

  await db
    .from("import_batches")
    .update({
      status: needsAttention ? "needs_attention" : "completed",
      questions_extracted: drafts.length,
      finished_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    })
    .eq("id", batchId);
}

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: CORS_HEADERS });
  if (req.method !== "POST") return json({ error: "Method not allowed" }, 405);

  if (!ANTHROPIC_API_KEY) {
    return json({ error: "ANTHROPIC_API_KEY is not configured for this project" }, 500);
  }

  const authHeader = req.headers.get("Authorization");
  if (!authHeader) return json({ error: "Missing Authorization header" }, 401);

  const callerClient = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
    global: { headers: { Authorization: authHeader } },
  });
  const { data: userData, error: userErr } = await callerClient.auth.getUser();
  if (userErr || !userData?.user) return json({ error: "Invalid session" }, 401);

  const { data: isAdmin, error: adminErr } = await callerClient.rpc("is_admin");
  if (adminErr || !isAdmin) return json({ error: "Admin access required" }, 403);

  let payload: { batch_id?: string; redo_draft_id?: number };
  try {
    payload = await req.json();
  } catch {
    return json({ error: "Invalid JSON body" }, 400);
  }

  const db = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

  // Redo path -- re-solves exactly one already-inserted draft_questions row
  // in place (e.g. to fix a bad LaTeX delimiter or an answer-key mismatch)
  // without touching the rest of the batch. Independent of the batch's
  // overall status, since the batch may already be 'completed'.
  if (payload.redo_draft_id) {
    const { data: draft, error: draftErr } = await db
      .from("draft_questions")
      .select("*")
      .eq("id", payload.redo_draft_id)
      .single();
    if (draftErr || !draft) return json({ error: `Draft question not found: ${draftErr?.message}` }, 404);

    const { data: batch, error: batchErr } = await db
      .from("import_batches")
      .select("*")
      .eq("id", draft.batch_id)
      .single();
    if (batchErr || !batch) return json({ error: `Import batch not found: ${batchErr?.message}` }, 404);

    const boundaries = (batch.boundaries_json ?? []) as QuestionBoundary[];
    const q = boundaries.find((b) => b.original_question_number === draft.original_question_number);
    if (!q) return json({ error: "Original question data is no longer on this batch -- cannot redo automatically." }, 404);

    const answerKeyMap = batch.answer_key ? parseAnswerKey(batch.answer_key as string) : new Map<number, string>();
    const hasAnswerKey = !!batch.answer_key;

    let dedupKeys = new Set<string>();
    if (batch.original_test) {
      const { data: existingQs } = await db
        .from("questions")
        .select("original_question_number")
        .eq("original_test", batch.original_test);
      dedupKeys = new Set((existingQs ?? []).map((r) => `${batch.original_test}|${r.original_question_number}`));
    }

    const pagePdfs = q.needs_image ? await loadPagePdfs(db, batch.source_pdf_path ?? null) : [];
    const fields = await solveAndBuildFields(q, pagePdfs, batch.original_test ?? null, answerKeyMap, hasAnswerKey, dedupKeys);

    const { error: updateErr } = await db.from("draft_questions").update(fields).eq("id", draft.id);
    if (updateErr) return json({ error: `Failed to update question: ${updateErr.message}` }, 500);

    return json({ redone: true, draft_id: draft.id, ...fields }, 200);
  }

  const { batch_id } = payload;
  if (!batch_id) return json({ error: "batch_id is required" }, 400);

  // Wrapped end-to-end -- without this, any unexpected throw (a constraint
  // violation on insert, a transient DB error, etc.) crashes the function
  // uncaught and the client only ever sees a generic "non-2xx status code"
  // with no real message, since Deno's default error response carries no
  // JSON `error` field for the client to read.
  try {
    const { data: batch, error: batchErr } = await db
      .from("import_batches")
      .select("*")
      .eq("id", batch_id)
      .single();
    if (batchErr || !batch) return json({ error: `Import batch not found: ${batchErr?.message}` }, 404);

    if (batch.status !== "transcribed" && batch.status !== "processing") {
      return json({ error: `Batch is in status '${batch.status}'. Nothing to solve.` }, 409);
    }
    const boundaries = (batch.boundaries_json ?? []) as QuestionBoundary[];
    if (boundaries.length === 0) {
      return json({ error: "Batch has no transcribed questions to solve" }, 409);
    }

    const isFirstCall = batch.status === "transcribed";
    if (isFirstCall) {
      await db.from("import_batches").update({ status: "processing" }).eq("id", batch_id);
    }

    const { data: existingDrafts } = await db
      .from("draft_questions")
      .select("original_question_number")
      .eq("batch_id", batch_id);
    const solvedNumbers = new Set((existingDrafts ?? []).map((r) => r.original_question_number));
    const remaining = boundaries.filter((b) => !solvedNumbers.has(b.original_question_number));
    const remainingTextOnly = remaining.filter((b) => !b.needs_image);
    const remainingNeedsImage = remaining.filter((b) => b.needs_image);
    const totalNeedsImage = boundaries.filter((b) => b.needs_image).length;

    const answerKeyMap = batch.answer_key ? parseAnswerKey(batch.answer_key as string) : new Map<number, string>();
    const hasAnswerKey = !!batch.answer_key;

    // Dedup check against the published bank -- scoped to this batch's
    // original_test so it stays a single cheap query.
    let dedupKeys = new Set<string>();
    if (batch.original_test) {
      const { data: existingQs } = await db
        .from("questions")
        .select("original_question_number")
        .eq("original_test", batch.original_test);
      dedupKeys = new Set((existingQs ?? []).map((r) => `${batch.original_test}|${r.original_question_number}`));
    }

    if (isFirstCall && remainingTextOnly.length > 0) {
      // @ts-ignore -- EdgeRuntime is a Supabase Edge Functions global, typed by the
      // jsr:@supabase/functions-js/edge-runtime.d.ts import at the top of this file.
      EdgeRuntime.waitUntil(
        solveTextOnlyBatch(
          db, batch_id, remainingTextOnly,
          batch.source_label ?? null, batch.original_test ?? null,
          answerKeyMap, hasAnswerKey, dedupKeys,
        ),
      );
    }

    if (remainingNeedsImage.length > 0) {
      const pagePdfs = await loadPagePdfs(db, batch.source_pdf_path ?? null);
      await solveAndInsertOne(
        db, batch_id, remainingNeedsImage[0], pagePdfs,
        batch.source_label ?? null, batch.original_test ?? null,
        answerKeyMap, hasAnswerKey, dedupKeys,
      );
    }

    await finalizeIfDone(db, batch_id);

    // Heartbeat -- needs_image questions are solved one per call with a paced
    // delay between them and don't otherwise touch import_batches, so without
    // this a long multi-image-question batch could go quiet long enough to
    // look stale even while genuinely progressing.
    await db.from("import_batches").update({ updated_at: new Date().toISOString() }).eq("id", batch_id);

    const { data: freshBatch } = await db
      .from("import_batches")
      .select("status, error_message")
      .eq("id", batch_id)
      .single();
    const { count: solvedCount } = await db
      .from("draft_questions")
      .select("*", { count: "exact", head: true })
      .eq("batch_id", batch_id);

    return json({
      batch_id,
      status: freshBatch?.status ?? "processing",
      error_message: freshBatch?.error_message ?? null,
      needs_image_total: totalNeedsImage,
      needs_image_remaining: Math.max(0, remainingNeedsImage.length - 1),
      questions_solved: solvedCount ?? 0,
    }, 202);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    await markFailed(db, batch_id, message);
    return json({ error: message }, 500);
  }
});

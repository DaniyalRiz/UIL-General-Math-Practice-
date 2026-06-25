// Admin-only: Step 2 of the PDF import pipeline. Takes a batch_id whose
// import_batches row already has status='transcribed' (written by
// import-test-pdf / step 1) and reads `boundaries_json` back out of the
// database -- not from any in-memory state -- to solve each question and
// write draft_questions rows for human review. Never touches `questions` and
// never sets review_status to anything but 'pending'.
import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { createClient } from "jsr:@supabase/supabase-js@2";
import { encodeBase64 } from "jsr:@std/encoding/base64";
import { PDFDocument } from "npm:pdf-lib@1.17.1";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const ANTHROPIC_API_KEY = Deno.env.get("ANTHROPIC_API_KEY");
const ANTHROPIC_MODEL = "claude-sonnet-4-6";
const CALL_TIMEOUT_MS = 140_000;

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

// Solves exactly one already-transcribed question.
const SOLVE_TOOL = {
  name: "solve_question",
  description: "Solve this single math competition question and show the full worked solution.",
  strict: true,
  input_schema: {
    type: "object",
    properties: {
      extracted_answer: { type: "string", description: "The correct choice, exactly matching one entry in choices" },
      explanation: { type: "string", description: "Full worked solution in LaTeX" },
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

// Streams a Messages API tool-forced response and returns the assembled tool
// input, instead of waiting for one blocking JSON reply. Raw SSE parsing (no
// SDK), matching the rest of this function's fetch-based style.
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
// against the Anthropic API small instead of firing everything at once.
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
    .update({ status: "failed", error_message: message, finished_at: new Date().toISOString() })
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
    content.push({
      type: "document",
      source: { type: "base64", media_type: "application/pdf", data: pagePdfs[q._pageIndex] },
    });
  }
  content.push({
    type: "text",
    text:
      `Solve this math competition question and show your real work.\n\n` +
      `Question ${q.original_question_number}: ${q.question}\n\n` +
      `Choices:\n${q.choices.join("\n")}\n\n` +
      (q.needs_image && pagePdfs[q._pageIndex] ? "The PDF page is attached above because this question depends on a diagram -- look at it carefully.\n" : "") +
      `Set extracted_answer to exactly one of the choices above, and explanation to the full worked solution in LaTeX.`,
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

// Runs after the kickoff response has already been sent to the browser. Reads
// boundaries_json + answer_key + original_test back out of the batch row
// (not from any earlier in-memory state), solves each question independently
// and in parallel, inserting each draft_questions row as soon as it's ready
// so the review UI can show partial results live.
async function runSolving(
  db: SupabaseClient,
  batchId: string,
  boundaries: QuestionBoundary[],
  sourceLabel: string | null,
  originalTest: string | null,
  answerKey: string | null,
  sourcePdfPath: string | null,
) {
  const pagePdfs = await loadPagePdfs(db, sourcePdfPath);
  const answerKeyMap = answerKey ? parseAnswerKey(answerKey) : new Map<number, string>();

  const results = await runWithConcurrency(boundaries, 6, async (q) => {
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
    let verification_status: "match" | "mismatch" | "unverified" | "no_answer_key" = "unverified";
    let verification_notes: string | null = solveError ? `Solving failed: ${solveError}` : null;
    if (solveError) {
      verification_status = "unverified";
    } else if (!answerKey) {
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

    const matchedChoice = q.choices.find((c) => choiceLetter(c) === (keyLetter ?? extractedLetter));

    const row = {
      batch_id: batchId,
      title: q.title,
      topic: q.topic,
      difficulty: q.difficulty,
      source: sourceLabel ?? null,
      question: q.question,
      choices: q.choices,
      tags: q.tags ?? [],
      extracted_answer,
      claude_solved_answer: extracted_answer,
      answer: matchedChoice ?? extracted_answer,
      explanation,
      needs_image: q.needs_image,
      image_alt: q.needs_image ? q.image_alt : null,
      verification_status,
      verification_notes,
      original_test: originalTest ?? null,
      original_question_number: q.original_question_number,
      source_reference: sourceLabel ? `${sourceLabel} #${q.original_question_number}` : null,
      review_status: "pending",
    };

    // Insert immediately -- this is what makes results show up live in the
    // review UI instead of waiting for every question to finish.
    const { error: insertErr } = await db.from("draft_questions").insert(row);
    if (insertErr) throw new Error(`Failed to insert question ${q.original_question_number}: ${insertErr.message}`);
    return { verification_status, hadError: !!solveError };
  });

  const inserted = results.filter((r) => r.status === "fulfilled").length;
  const needsAttention = results.some(
    (r) => r.status === "rejected" || (r.status === "fulfilled" && (r.value.verification_status === "mismatch" || r.value.hadError)),
  );

  if (inserted === 0) {
    await markFailed(db, batchId, "All questions failed to solve -- check individual question errors and retry.");
    return;
  }

  await db
    .from("import_batches")
    .update({
      status: needsAttention ? "needs_attention" : "completed",
      questions_extracted: inserted,
      finished_at: new Date().toISOString(),
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

  let payload: { batch_id?: string };
  try {
    payload = await req.json();
  } catch {
    return json({ error: "Invalid JSON body" }, 400);
  }

  const { batch_id } = payload;
  if (!batch_id) return json({ error: "batch_id is required" }, 400);

  const db = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

  const { data: batch, error: batchErr } = await db
    .from("import_batches")
    .select("*")
    .eq("id", batch_id)
    .single();
  if (batchErr || !batch) return json({ error: `Import batch not found: ${batchErr?.message}` }, 404);

  if (batch.status !== "transcribed") {
    return json({ error: `Batch is in status '${batch.status}', expected 'transcribed'. Nothing to solve.` }, 409);
  }
  if (!batch.boundaries_json || !Array.isArray(batch.boundaries_json) || batch.boundaries_json.length === 0) {
    return json({ error: "Batch has no transcribed questions to solve" }, 409);
  }

  await db.from("import_batches").update({ status: "processing" }).eq("id", batch_id);

  // @ts-ignore -- EdgeRuntime is a Supabase Edge Functions global, typed by the
  // jsr:@supabase/functions-js/edge-runtime.d.ts import at the top of this file.
  EdgeRuntime.waitUntil(
    runSolving(
      db,
      batch_id,
      batch.boundaries_json as QuestionBoundary[],
      batch.source_label ?? null,
      batch.original_test ?? null,
      batch.answer_key ?? null,
      batch.source_pdf_path ?? null,
    ),
  );

  return json({ batch_id, status: "processing" }, 202);
});

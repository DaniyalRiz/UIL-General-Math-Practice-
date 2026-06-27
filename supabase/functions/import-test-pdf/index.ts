// Admin-only: Step 1 of the PDF import pipeline. Transcribes one page per
// invocation into JSON and persists it durably on the import_batches row
// (boundaries_json, appended page by page; status -> 'transcribed' once
// every page is done). Step 2 (solve-pdf-questions) reads that JSON back out
// of the database to generate answers/explanations, so the two steps are
// independently resumable.
//
// One page per call, synchronously (no EdgeRuntime.waitUntil) -- a single
// page's PDF can already use most of a Sonnet-tier 30,000-input-tokens/
// minute budget (confirmed empirically: splitting the PDF into one-page
// files did NOT shrink the actual Claude-billed size as much as expected,
// since pdf-lib re-embeds each page's full font/resource set into every
// single-page copy). Transcription runs on Haiku specifically because
// Anthropic's per-minute rate limit is scoped per model -- Haiku gets its
// own separate, much larger budget, so it sidesteps the Sonnet bottleneck
// entirely rather than just pacing around it. Solving (step 2) stays on
// Sonnet, since that's actual math reasoning, not mechanical transcription.
// The browser still drives this forward with a delay between calls as a
// safety margin, using next_page_index/pages_total from each response to
// know when to stop. Never touches `questions` and never sets
// review_status to anything but 'pending'.
import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { createClient } from "jsr:@supabase/supabase-js@2";
import { encodeBase64 } from "jsr:@std/encoding/base64";
import { PDFDocument } from "npm:pdf-lib@1.17.1";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const ANTHROPIC_API_KEY = Deno.env.get("ANTHROPIC_API_KEY");
const ANTHROPIC_MODEL = "claude-haiku-4-5-20251001";
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

const BOUNDARIES_TOOL = {
  name: "extract_question_boundaries",
  description: "Transcribe every multiple-choice question from this math competition test page, without solving them, and flag if this page is an answer key.",
  strict: true,
  input_schema: {
    type: "object",
    properties: {
      questions: {
        type: "array",
        items: {
          type: "object",
          properties: {
            original_question_number: { type: "integer", description: "Question number as printed on the test" },
            title: { type: "string", description: "Short descriptive title, 3-8 words" },
            topic: { type: "string", description: "e.g. Algebra 1 & 2, Geometry, Precalculus, Number Sense" },
            difficulty: { type: "string", enum: ["Easy", "Medium", "Hard"] },
            question: { type: "string", description: "Full question text, transcribed exactly. Use LaTeX (\\(...\\)) for math notation." },
            choices: {
              type: "array",
              items: { type: "string" },
              description: "Exactly 5 answer choices, each formatted like '(A) 42'",
            },
            tags: { type: "array", items: { type: "string" } },
            needs_image: { type: "boolean", description: "True if a diagram/figure is essential to solving the question, not just decorative" },
            image_alt: { type: "string", description: "Alt text describing the needed diagram, empty string if needs_image is false" },
          },
          required: [
            "original_question_number", "title", "topic", "difficulty", "question",
            "choices", "tags", "needs_image", "image_alt",
          ],
          additionalProperties: false,
        },
      },
      answer_key_text: {
        type: ["string", "null"],
        description:
          "If this page is an answer key (a list of question numbers with their correct letter, often near the end " +
          "of the test, e.g. '1. C  2. D  3. A ...'), the raw text of that list exactly as printed. Null otherwise.",
      },
    },
    required: ["questions", "answer_key_text"],
    additionalProperties: false,
  },
};

type ClaudeQuestionBoundary = {
  original_question_number: number;
  title: string;
  topic: string;
  difficulty: string;
  question: string;
  choices: string[];
  tags: string[];
  needs_image: boolean;
  image_alt: string;
};

// _pageIndex is set by our own code (which page-PDF this came from), never by
// Claude -- step 2 uses it to re-attach just that one page for needs_image
// questions instead of the whole document.
type QuestionBoundary = ClaudeQuestionBoundary & { _pageIndex: number };

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
    throw new Error(`[${label}] Claude API error ${resp.status}: ${errText}`);
  }
  if (!resp.body) throw new Error(`[${label}] Claude API returned no response body for the streamed request`);

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let partialJson = "";
  let stopReason = "end_turn";
  let parseFailures = 0;
  const eventTypesSeen: string[] = [];

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
      let evt: { type?: string; delta?: { type?: string; partial_json?: string; stop_reason?: string }; content_block?: { type?: string } };
      try {
        evt = JSON.parse(dataStr);
      } catch {
        parseFailures++;
        continue;
      }
      if (evt.type && eventTypesSeen.length < 20) {
        eventTypesSeen.push(evt.delta?.type ? `${evt.type}:${evt.delta.type}` : evt.type);
      }
      if (evt.type === "content_block_delta" && evt.delta?.type === "input_json_delta") {
        partialJson += evt.delta.partial_json ?? "";
      } else if (evt.type === "message_delta" && evt.delta?.stop_reason) {
        stopReason = evt.delta.stop_reason;
      }
    }
  }

  if (!partialJson) {
    throw new Error(
      `[${label}] Claude did not stream any tool input. stop_reason=${stopReason}, parseFailures=${parseFailures}, ` +
        `partialJsonLen=${partialJson.length}, eventTypes=${JSON.stringify(eventTypesSeen)}`,
    );
  }

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

type SupabaseClient = ReturnType<typeof createClient>;

async function markFailed(db: SupabaseClient, batchId: string, message: string) {
  await db
    .from("import_batches")
    .update({ status: "failed", error_message: message, finished_at: new Date().toISOString(), updated_at: new Date().toISOString() })
    .eq("id", batchId);
}

async function extractBoundariesForPage(
  pagePdfBase64: string,
  pageIndex: number,
): Promise<{ questions: QuestionBoundary[]; answerKeyText: string | null }> {
  const { stopReason, toolInput } = await withTimeout((signal) =>
    streamToolCall(
      {
        model: ANTHROPIC_MODEL,
        max_tokens: 6000,
        stream: true,
        tools: [BOUNDARIES_TOOL],
        tool_choice: { type: "tool", name: "extract_question_boundaries" },
        messages: [
          {
            role: "user",
            content: [
              { type: "document", source: { type: "base64", media_type: "application/pdf", data: pagePdfBase64 } },
              {
                type: "text",
                text:
                  "Transcribe every multiple-choice math question on this page exactly as printed. Do not solve them " +
                  "yet. Preserve LaTeX-worthy math notation using \\(...\\) inline math. Set needs_image:true only " +
                  "when a diagram is essential to solving the question, not just decorative. If this page has no " +
                  "multiple-choice math questions on it (e.g. it is a cover page, instructions, a blank/filler page, " +
                  "or worked solutions), return an empty questions array. If this page IS an answer key -- a list of " +
                  "question numbers with their correct letter, typically near the end of the test -- set " +
                  "answer_key_text to that list's raw text exactly as printed (questions array still empty for this " +
                  "page); this is important, the answer key must not be discarded. Otherwise set answer_key_text to null.",
              },
            ],
          },
        ],
      },
      signal,
      `boundaries-p${pageIndex + 1}`,
    )
  );

  if (stopReason === "refusal") throw new Error(`Claude declined to process page ${pageIndex + 1} (refusal)`);
  const raw = (toolInput.questions as ClaudeQuestionBoundary[] | undefined) ?? [];
  const answerKeyText = (toolInput.answer_key_text as string | null | undefined) ?? null;
  return { questions: raw.map((q) => ({ ...q, _pageIndex: pageIndex })), answerKeyText };
}

// Re-downloads the original PDF (or uses the freshly-uploaded bytes on the
// very first call, to avoid an extra round-trip), isolates exactly the one
// page at `pageIndex` via pdf-lib, and transcribes it. Persists the combined
// JSON directly on the batch row after every single page -- this is "the
// JSON file": durable, inspectable, and exactly what step 2 reads.
async function processNextPage(
  db: SupabaseClient,
  batchId: string,
  pdfBytes: Uint8Array | null,
  pagesTotalHint: number | null,
  pageIndex: number,
) {
  let bytes = pdfBytes;
  if (!bytes) {
    const { data: batch } = await db.from("import_batches").select("source_pdf_path").eq("id", batchId).single();
    const path = batch?.source_pdf_path as string | undefined;
    if (!path) throw new Error("Original PDF is not available in storage to continue transcription");
    const { data, error } = await db.storage.from("test-pdfs").download(path);
    if (error || !data) throw new Error(`Could not re-download source PDF: ${error?.message}`);
    bytes = new Uint8Array(await data.arrayBuffer());
  }

  const srcDoc = await PDFDocument.load(bytes);
  const pageCount = pagesTotalHint ?? srcDoc.getPageCount();
  const newDoc = await PDFDocument.create();
  const [copiedPage] = await newDoc.copyPages(srcDoc, [pageIndex]);
  newDoc.addPage(copiedPage);
  const pagePdfBase64 = encodeBase64(await newDoc.save());

  let newQuestions: QuestionBoundary[] = [];
  let detectedAnswerKey: string | null = null;
  let pageError: string | null = null;
  try {
    const result = await extractBoundariesForPage(pagePdfBase64, pageIndex);
    newQuestions = result.questions;
    detectedAnswerKey = result.answerKeyText;
  } catch (err) {
    pageError = err instanceof Error && err.name === "AbortError"
      ? `page ${pageIndex + 1}: timed out after ${CALL_TIMEOUT_MS / 1000}s`
      : `page ${pageIndex + 1}: ${err instanceof Error ? err.message : String(err)}`;
  }

  const { data: current } = await db
    .from("import_batches")
    .select("boundaries_json, error_message, answer_key")
    .eq("id", batchId)
    .single();
  const boundaries = [...((current?.boundaries_json ?? []) as QuestionBoundary[]), ...newQuestions];
  const priorError = (current?.error_message as string | null) ?? null;
  const combinedError = pageError ? (priorError ? `${priorError} | ${pageError}` : pageError) : priorError;

  // A key typed in manually by the admin always wins -- this only fills the
  // gap when the test's own printed answer key would otherwise be discarded
  // (transcription ignores answer-key pages, so step 2 never saw it before).
  const existingAnswerKey = (current?.answer_key as string | null) ?? null;
  const answerKey = existingAnswerKey || detectedAnswerKey;

  const nextPageIndex = pageIndex + 1;
  const isDone = nextPageIndex >= pageCount;

  if (isDone && boundaries.length === 0) {
    const message = combinedError ?? "No multiple-choice questions found anywhere in this PDF.";
    await markFailed(db, batchId, message);
    return { batch_id: batchId, status: "failed", pages_total: pageCount, next_page_index: nextPageIndex, questions_so_far: 0, error_message: message };
  }

  await db
    .from("import_batches")
    .update({
      status: isDone ? "transcribed" : "processing",
      boundaries_json: boundaries,
      next_page_index: nextPageIndex,
      questions_total: boundaries.length,
      error_message: combinedError,
      answer_key: answerKey,
      answer_key_found: !!answerKey,
      updated_at: new Date().toISOString(),
    })
    .eq("id", batchId);

  return {
    batch_id: batchId,
    status: isDone ? "transcribed" : "processing",
    pages_total: pageCount,
    next_page_index: nextPageIndex,
    questions_so_far: boundaries.length,
    error_message: combinedError,
    answer_key_found: !!answerKey,
  };
}

async function startBatch(
  db: SupabaseClient,
  userId: string | null,
  pdf_base64: string,
  source_label: string | null,
  original_test: string | null,
  answer_key: string | null,
) {
  const { data: batch, error: batchErr } = await db
    .from("import_batches")
    .insert({
      created_by: userId,
      source_label,
      original_test,
      answer_key,
      status: "processing",
      started_at: new Date().toISOString(),
      answer_key_found: !!answer_key,
      boundaries_json: [],
      next_page_index: 0,
    })
    .select()
    .single();
  if (batchErr || !batch) throw new Error(`Failed to create import batch: ${batchErr?.message}`);

  const bytes = Uint8Array.from(atob(pdf_base64), (c) => c.charCodeAt(0));

  // Best-effort audit copy -- needed so later continuation calls (and step 2)
  // can re-download the original PDF, since each invocation is stateless.
  // A failure here still must not block transcription itself.
  try {
    const path = `${batch.id}.pdf`;
    await db.storage.from("test-pdfs").upload(path, bytes, { contentType: "application/pdf", upsert: true });
    await db.from("import_batches").update({ source_pdf_path: path }).eq("id", batch.id);
  } catch {
    // Non-fatal.
  }

  let pageCount: number;
  try {
    pageCount = (await PDFDocument.load(bytes)).getPageCount();
  } catch {
    pageCount = 1;
  }
  await db.from("import_batches").update({ pages_total: pageCount }).eq("id", batch.id);

  return processNextPage(db, batch.id, bytes, pageCount, 0);
}

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: CORS_HEADERS });
  if (req.method !== "POST") return json({ error: "Method not allowed" }, 405);

  if (!ANTHROPIC_API_KEY) {
    return json({ error: "ANTHROPIC_API_KEY is not configured for this project" }, 500);
  }

  // Service role for the actual writes -- RLS on these tables is admin-only anyway,
  // but we verify admin status (or the cron secret) below before doing anything.
  const db = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

  // Two ways to be authorized: an interactive admin session (browser), or
  // the cron tick's shared secret (no admin session exists for a
  // server-triggered call). The secret itself is never embedded in this
  // file -- it's fetched from Vault at request time and compared. The cron
  // path only ever sends a batch_id (it has no PDF bytes to upload), so it
  // naturally can't reach the fresh-upload branch below.
  const authHeader = req.headers.get("Authorization");
  const cronSecretHeader = req.headers.get("x-cron-secret");
  let isTrustedCron = false;
  if (cronSecretHeader) {
    const { data: expectedSecret } = await db.rpc("internal_get_secret", { p_name: "cron_internal_secret" });
    isTrustedCron = !!expectedSecret && cronSecretHeader === expectedSecret;
  }

  let userId: string | null = null;
  if (!isTrustedCron) {
    if (!authHeader) return json({ error: "Missing Authorization header" }, 401);

    // Verify the caller is an admin using THEIR OWN jwt -- never trust a client-claimed role.
    const callerClient = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
      global: { headers: { Authorization: authHeader } },
    });
    const { data: userData, error: userErr } = await callerClient.auth.getUser();
    if (userErr || !userData?.user) return json({ error: "Invalid session" }, 401);

    const { data: isAdmin, error: adminErr } = await callerClient.rpc("is_admin");
    if (adminErr || !isAdmin) return json({ error: "Admin access required" }, 403);
    userId = userData.user.id;
  }

  let payload: {
    pdf_base64?: string;
    source_label?: string;
    original_test?: string;
    answer_key?: string;
    batch_id?: string;
  };
  try {
    payload = await req.json();
  } catch {
    return json({ error: "Invalid JSON body" }, 400);
  }

  if (payload.batch_id) {
    // Continuation call -- transcribe the next page of an already-started
    // batch. Claimed via the same lease as solve-pdf-questions, so the
    // browser's poll loop and a cron tick (or two overlapping cron ticks)
    // can never both transcribe the same page and pay for it twice.
    let leaseHeld = true;
    try {
      const { data: batch, error: claimErr } = await db.rpc("claim_import_batch", { p_batch_id: payload.batch_id, p_lease_minutes: 3 });
      if (claimErr) return json({ error: `Failed to claim batch: ${claimErr.message}` }, 500);
      if (!batch) {
        leaseHeld = false;
        const { data: exists } = await db.from("import_batches").select("id").eq("id", payload.batch_id).maybeSingle();
        if (!exists) return json({ error: `Import batch not found: ${payload.batch_id}` }, 404);
        return json({ error: "This batch is currently being processed by another request. Try again shortly." }, 409);
      }
      if (batch.status !== "processing") {
        return json({ error: `Batch is in status '${batch.status}'. Nothing to transcribe.` }, 409);
      }
      if (batch.next_page_index >= (batch.pages_total ?? 0)) {
        return json({ error: "This batch has already transcribed every page." }, 409);
      }
      const result = await processNextPage(db, batch.id, null, batch.pages_total, batch.next_page_index);
      return json(result, 202);
    } catch (err) {
      return json({ error: err instanceof Error ? err.message : String(err) }, 500);
    } finally {
      if (leaseHeld) {
        await db.from("import_batches").update({ lease_until: null }).eq("id", payload.batch_id);
      }
    }
  }

  try {
    // First call -- create the batch and transcribe page 0. Browser-only:
    // the cron path never reaches here since it has no PDF bytes to upload,
    // only ever continuing an existing batch via batch_id above.
    const { pdf_base64, source_label, original_test, answer_key } = payload;
    if (!pdf_base64) return json({ error: "pdf_base64 is required" }, 400);

    // Cost guardrail: refuse a new import while another one is still active,
    // so an accidental double-submit can't double the API spend for nothing.
    const { data: active } = await db
      .from("import_batches")
      .select("id, source_label")
      .in("status", ["queued", "processing"])
      .limit(1);
    if (active && active.length > 0) {
      return json(
        { error: `An import is already in progress (${active[0].source_label ?? active[0].id}). Wait for it to finish first.` },
        409,
      );
    }

    const result = await startBatch(db, userId, pdf_base64, source_label ?? null, original_test ?? null, answer_key ?? null);
    return json(result, 202);
  } catch (err) {
    return json({ error: err instanceof Error ? err.message : String(err) }, 500);
  }
});

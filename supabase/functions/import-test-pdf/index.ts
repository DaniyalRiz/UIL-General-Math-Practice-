// Admin-only: Step 1 of the PDF import pipeline. Transcribes every question
// on a UIL/TMSCA test PDF into JSON (no solving) and persists it durably on
// the import_batches row (status -> 'transcribed'). Step 2 (solve-pdf-questions)
// reads that JSON back out of the database to generate answers/explanations,
// so the two steps are independently resumable -- if solving fails or needs a
// retry, the PDF never has to be re-read (and re-rate-limited) again.
// Never touches `questions` and never sets review_status to anything but 'pending'.
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

// Transcription tool -- no solving, so this stays fast no matter how many
// questions are on the page.
const BOUNDARIES_TOOL = {
  name: "extract_question_boundaries",
  description: "Transcribe every multiple-choice question from this math competition test page, without solving them.",
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
    },
    required: ["questions"],
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

// Runs `fn` over every item like Promise.allSettled, but never more than
// `limit` calls in flight at once. Firing all page calls at the same instant
// spikes both request rate and token throughput against the Anthropic API and
// reliably trips rate limits -- this keeps the burst small while still
// running mostly in parallel.
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

// Physically splits the PDF into one minimal single-page PDF per page (vector
// copy via pdf-lib, no rasterization). The account's rate limit is 30,000
// input tokens/minute, and the whole multi-page document alone can already be
// ~30k+ tokens, so sending the full PDF on every per-page call was guaranteed
// to trip the limit. Splitting first means each call's input is roughly
// 1/pageCount the size, comfortably leaving room for several calls per minute.
async function splitPdfIntoPages(pdf_base64: string): Promise<string[]> {
  const bytes = Uint8Array.from(atob(pdf_base64), (c) => c.charCodeAt(0));
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
}

// Transcribes only the questions on one already-isolated single page.
async function extractBoundariesForPage(
  pagePdfBase64: string,
  pageIndex: number,
): Promise<QuestionBoundary[]> {
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
                  "an answer key, or worked solutions), return an empty questions array.",
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
  return raw.map((q) => ({ ...q, _pageIndex: pageIndex }));
}

// Runs after the kickoff response has already been sent to the browser (see
// EdgeRuntime.waitUntil below). Splits the PDF into pages, transcribes each
// page in parallel (concurrency-limited), and persists the combined JSON
// directly on the batch row -- this is "the JSON file": durable, inspectable,
// and exactly what step 2 (solve-pdf-questions) reads to generate answers.
async function runTranscription(db: SupabaseClient, batchId: string, pdf_base64: string) {
  await db.from("import_batches").update({ status: "processing" }).eq("id", batchId);

  // Best-effort audit copy -- step 2 re-downloads this to recover per-page
  // PDFs for needs_image questions, so it's not purely cosmetic anymore, but
  // a failure here still must not block transcription itself.
  try {
    const bytes = Uint8Array.from(atob(pdf_base64), (c) => c.charCodeAt(0));
    const path = `${batchId}.pdf`;
    await db.storage.from("test-pdfs").upload(path, bytes, { contentType: "application/pdf", upsert: true });
    await db.from("import_batches").update({ source_pdf_path: path }).eq("id", batchId);
  } catch {
    // Non-fatal -- proceed with transcription regardless.
  }

  let pagePdfs: string[];
  try {
    pagePdfs = await splitPdfIntoPages(pdf_base64);
  } catch {
    // pdf-lib couldn't parse this file (e.g. encrypted/corrupted) -- fall back
    // to treating it as one page. Works fine for small PDFs, may still hit the
    // rate limit on a large one, but that's strictly better than hard-failing.
    pagePdfs = [pdf_base64];
  }
  const pageCount = pagePdfs.length;
  const pageIndexes = Array.from({ length: pageCount }, (_, i) => i);

  // TEMPORARY diagnostic: confirm whether per-page splitting actually shrank
  // each page's byte size, or whether pdf-lib re-embedded a large shared
  // font/resource into every single-page copy (a known pdf-lib behavior),
  // which would explain still hitting the 30k-tokens/minute rate limit.
  const pageSizesKB = pagePdfs.map((p, i) => `p${i + 1}:${Math.round((p.length * 0.75) / 1024)}KB`).join(", ");

  const pageResults = await runWithConcurrency(
    pageIndexes,
    4,
    (pageIndex) => extractBoundariesForPage(pagePdfs[pageIndex], pageIndex),
  );

  const boundaries: QuestionBoundary[] = [];
  const pageErrors: string[] = [];
  pageResults.forEach((r, idx) => {
    if (r.status === "fulfilled") {
      boundaries.push(...r.value);
    } else {
      const reason = r.reason instanceof Error && r.reason.name === "AbortError"
        ? `timed out after ${CALL_TIMEOUT_MS / 1000}s`
        : r.reason instanceof Error
          ? r.reason.message
          : String(r.reason);
      pageErrors.push(`page ${idx + 1}: ${reason}`);
    }
  });

  if (boundaries.length === 0) {
    await markFailed(
      db,
      batchId,
      `[page sizes: ${pageSizesKB}] ` +
        (pageErrors.length > 0
          ? `No questions transcribed from any page. Per-page errors: ${pageErrors.join(" | ")}`
          : "No multiple-choice questions found anywhere in this PDF."),
    );
    return;
  }

  await db
    .from("import_batches")
    .update({
      status: "transcribed",
      boundaries_json: boundaries,
      questions_total: boundaries.length,
      error_message: `[page sizes: ${pageSizesKB}]` + (pageErrors.length > 0
        ? ` ${pageErrors.length} of ${pageCount} page(s) failed to transcribe and are missing from this import: ${pageErrors.join(" | ")}`
        : ""),
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

  // Verify the caller is an admin using THEIR OWN jwt -- never trust a client-claimed role.
  const callerClient = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
    global: { headers: { Authorization: authHeader } },
  });
  const { data: userData, error: userErr } = await callerClient.auth.getUser();
  if (userErr || !userData?.user) return json({ error: "Invalid session" }, 401);

  const { data: isAdmin, error: adminErr } = await callerClient.rpc("is_admin");
  if (adminErr || !isAdmin) return json({ error: "Admin access required" }, 403);

  let payload: {
    pdf_base64?: string;
    source_label?: string;
    original_test?: string;
    answer_key?: string;
  };
  try {
    payload = await req.json();
  } catch {
    return json({ error: "Invalid JSON body" }, 400);
  }

  const { pdf_base64, source_label, original_test, answer_key } = payload;
  if (!pdf_base64) return json({ error: "pdf_base64 is required" }, 400);

  // Service role for the actual writes -- RLS on these tables is admin-only anyway,
  // but we've already verified admin status above against the caller's own session.
  const db = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

  const { data: batch, error: batchErr } = await db
    .from("import_batches")
    .insert({
      created_by: userData.user.id,
      source_label: source_label ?? null,
      original_test: original_test ?? null,
      answer_key: answer_key ?? null,
      status: "queued",
      started_at: new Date().toISOString(),
      answer_key_found: !!answer_key,
    })
    .select()
    .single();
  if (batchErr || !batch) return json({ error: `Failed to create import batch: ${batchErr?.message}` }, 500);

  // Respond immediately -- the browser polls import_batches for progress
  // instead of holding this request open. Transcription keeps running in the
  // background via EdgeRuntime.waitUntil, which is the documented way to do
  // work after the response has been sent without the isolate being torn
  // down early.
  // @ts-ignore -- EdgeRuntime is a Supabase Edge Functions global, typed by the
  // jsr:@supabase/functions-js/edge-runtime.d.ts import at the top of this file.
  EdgeRuntime.waitUntil(runTranscription(db, batch.id, pdf_base64));

  return json({ batch_id: batch.id, status: "queued" }, 202);
});

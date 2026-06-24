// Admin-only: extracts questions from a UIL/TMSCA test PDF via Claude and
// writes them to draft_questions for human review. Never touches `questions`
// and never sets review_status to anything but 'pending'.
import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { createClient } from "jsr:@supabase/supabase-js@2";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const ANTHROPIC_API_KEY = Deno.env.get("ANTHROPIC_API_KEY");
const ANTHROPIC_MODEL = "claude-sonnet-4-6";
// Worst-case safety bound on the streamed Claude call, not the expected
// completion path — streaming means we keep receiving good output the whole
// time, so this only fires for a genuinely hung request. Must stay below the
// reap_stale_import_batches() staleness threshold (6 minutes) so that safety
// net never preempts a call that's still actively streaming.
const CLAUDE_TIMEOUT_MS = 240_000;

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

const EXTRACT_TOOL = {
  name: "extract_questions",
  description: "Extract every multiple-choice question from this math competition test page.",
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
            question: { type: "string", description: "Full question text. Use LaTeX (\\(...\\)) for math notation." },
            choices: {
              type: "array",
              items: { type: "string" },
              description: "Exactly 5 answer choices, each formatted like '(A) 42'",
            },
            extracted_answer: { type: "string", description: "Best guess at the correct choice, exactly matching one entry in choices" },
            explanation: { type: "string", description: "Full worked solution in LaTeX" },
            tags: { type: "array", items: { type: "string" } },
            needs_image: { type: "boolean", description: "True if a diagram/figure is essential to understanding the question" },
            image_alt: { type: "string", description: "Alt text describing the needed diagram, empty string if needs_image is false" },
          },
          required: [
            "original_question_number", "title", "topic", "difficulty", "question",
            "choices", "extracted_answer", "explanation", "tags", "needs_image", "image_alt",
          ],
          additionalProperties: false,
        },
      },
    },
    required: ["questions"],
    additionalProperties: false,
  },
};

type ExtractedQuestion = {
  original_question_number: number;
  title: string;
  topic: string;
  difficulty: string;
  question: string;
  choices: string[];
  extracted_answer: string;
  explanation: string;
  tags: string[];
  needs_image: boolean;
  image_alt: string;
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

// Streams the Messages API response instead of waiting for one blocking JSON
// reply. Extracting+solving ~8 questions with full LaTeX worked solutions is
// a lot of output — streaming means we keep receiving real progress the whole
// time instead of guessing a single timeout long enough to cover worst case.
// Raw SSE parsing (no SDK), matching the rest of this function's fetch-based style.
async function streamExtraction(
  pdf_base64: string,
  signal: AbortSignal,
): Promise<{ stopReason: string; toolInput: { questions?: ExtractedQuestion[] } }> {
  const resp = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    signal,
    headers: {
      "content-type": "application/json",
      "x-api-key": ANTHROPIC_API_KEY!,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: ANTHROPIC_MODEL,
      max_tokens: 16000,
      stream: true,
      tools: [EXTRACT_TOOL],
      tool_choice: { type: "tool", name: "extract_questions" },
      messages: [
        {
          role: "user",
          content: [
            {
              type: "document",
              source: { type: "base64", media_type: "application/pdf", data: pdf_base64 },
            },
            {
              type: "text",
              text:
                "Extract every multiple-choice question from this math competition test page. " +
                "Solve each question yourself to determine extracted_answer — show your real work in explanation. " +
                "Preserve LaTeX-worthy math notation using \\(...\\) inline math. " +
                "Set needs_image:true only when a diagram is essential and not just decorative.",
            },
          ],
        },
      ],
    }),
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

  if (!partialJson) throw new Error("Claude did not stream any tool input");

  let toolInput: { questions?: ExtractedQuestion[] };
  try {
    toolInput = JSON.parse(partialJson);
  } catch (e) {
    throw new Error(`Failed to parse streamed tool input as JSON: ${e instanceof Error ? e.message : String(e)}`);
  }

  return { stopReason, toolInput };
}

type SupabaseClient = ReturnType<typeof createClient>;

async function markFailed(db: SupabaseClient, batchId: string, message: string) {
  await db
    .from("import_batches")
    .update({ status: "failed", error_message: message, finished_at: new Date().toISOString() })
    .eq("id", batchId);
}

// Runs after the kickoff response has already been sent to the browser (see
// EdgeRuntime.waitUntil below). Any failure here — including a Claude timeout —
// must end in a 'failed' row with a real error_message; never leave it hanging.
async function runExtraction(
  db: SupabaseClient,
  batchId: string,
  pdf_base64: string,
  source_label: string | null,
  original_test: string | null,
  answer_key: string | null,
) {
  await db.from("import_batches").update({ status: "processing" }).eq("id", batchId);

  // Best-effort audit copy in the bucket provisioned for this in Phase 1 — not
  // required for extraction (pdf_base64 is already in memory) so a failure here
  // must never block or fail the batch.
  try {
    const bytes = Uint8Array.from(atob(pdf_base64), (c) => c.charCodeAt(0));
    const path = `${batchId}.pdf`;
    await db.storage.from("test-pdfs").upload(path, bytes, { contentType: "application/pdf", upsert: true });
    await db.from("import_batches").update({ source_pdf_path: path }).eq("id", batchId);
  } catch {
    // Non-fatal — proceed with extraction regardless.
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), CLAUDE_TIMEOUT_MS);

  try {
    const { stopReason, toolInput } = await streamExtraction(pdf_base64, controller.signal);
    if (stopReason === "refusal") {
      throw new Error("Claude declined to process this PDF (refusal)");
    }

    const questions: ExtractedQuestion[] = toolInput.questions ?? [];
    if (questions.length === 0) throw new Error("No questions extracted from PDF");

    const answerKeyMap = answer_key ? parseAnswerKey(answer_key) : new Map<number, string>();

    const rows = questions.map((q) => {
      const extractedLetter = choiceLetter(q.extracted_answer) ?? choiceLetter(q.choices[0] ?? "");
      const keyLetter = answerKeyMap.get(q.original_question_number);
      let verification_status: "match" | "mismatch" | "unverified" | "no_answer_key" = "unverified";
      let verification_notes: string | null = null;
      if (!answer_key) {
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

      return {
        batch_id: batchId,
        title: q.title,
        topic: q.topic,
        difficulty: q.difficulty,
        source: source_label ?? null,
        question: q.question,
        choices: q.choices,
        tags: q.tags ?? [],
        extracted_answer: q.extracted_answer,
        claude_solved_answer: q.extracted_answer,
        answer: matchedChoice ?? q.extracted_answer,
        explanation: q.explanation,
        needs_image: q.needs_image,
        image_alt: q.needs_image ? q.image_alt : null,
        verification_status,
        verification_notes,
        original_test: original_test ?? null,
        original_question_number: q.original_question_number,
        source_reference: source_label ? `${source_label} #${q.original_question_number}` : null,
        review_status: "pending",
      };
    });

    const { error: insertErr } = await db.from("draft_questions").insert(rows);
    if (insertErr) throw new Error(`Failed to insert draft questions: ${insertErr.message}`);

    const needsAttention = rows.some((r) => r.verification_status === "mismatch");

    await db
      .from("import_batches")
      .update({
        status: needsAttention ? "needs_attention" : "completed",
        questions_extracted: rows.length,
        finished_at: new Date().toISOString(),
      })
      .eq("id", batchId);
  } catch (err) {
    const isTimeout = err instanceof Error && err.name === "AbortError";
    const message = isTimeout
      ? `Claude did not respond within ${CLAUDE_TIMEOUT_MS / 1000}s and the request was aborted. Try a page with fewer questions, or retry — long worked solutions on a dense page can exceed this budget.`
      : err instanceof Error
        ? err.message
        : String(err);
    await markFailed(db, batchId, message);
  } finally {
    clearTimeout(timeoutId);
  }
}

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: CORS_HEADERS });
  if (req.method !== "POST") return json({ error: "Method not allowed" }, 405);

  if (!ANTHROPIC_API_KEY) {
    return json({ error: "ANTHROPIC_API_KEY is not configured for this project" }, 500);
  }

  const authHeader = req.headers.get("Authorization");
  if (!authHeader) return json({ error: "Missing Authorization header" }, 401);

  // Verify the caller is an admin using THEIR OWN jwt — never trust a client-claimed role.
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

  // Service role for the actual writes — RLS on these tables is admin-only anyway,
  // but we've already verified admin status above against the caller's own session.
  const db = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

  const { data: batch, error: batchErr } = await db
    .from("import_batches")
    .insert({
      created_by: userData.user.id,
      source_label: source_label ?? null,
      status: "queued",
      started_at: new Date().toISOString(),
      answer_key_found: !!answer_key,
    })
    .select()
    .single();
  if (batchErr || !batch) return json({ error: `Failed to create import batch: ${batchErr?.message}` }, 500);

  // Respond immediately — the browser polls import_batches for status instead of
  // holding this request open. Extraction keeps running in the background via
  // EdgeRuntime.waitUntil, which is the documented way to do work after the
  // response has been sent without the isolate being torn down early.
  // @ts-ignore -- EdgeRuntime is a Supabase Edge Functions global, typed by the
  // jsr:@supabase/functions-js/edge-runtime.d.ts import at the top of this file.
  EdgeRuntime.waitUntil(
    runExtraction(db, batch.id, pdf_base64, source_label ?? null, original_test ?? null, answer_key ?? null),
  );

  return json({ batch_id: batch.id, status: "queued" }, 202);
});

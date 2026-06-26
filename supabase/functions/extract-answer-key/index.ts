// Admin-only: standalone helper for Review Imports. Transcribes a separate
// answer-key file (PDF or image, normally just one page) and returns the raw
// text directly -- no import_batches row, no pacing, since this is always
// one small file in one call. The returned text is meant to land in the
// existing manual "Answer key" field, which already takes priority over any
// answer key auto-detected from inside the main test PDF (import-test-pdf).
import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { createClient } from "jsr:@supabase/supabase-js@2";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY")!;
const ANTHROPIC_API_KEY = Deno.env.get("ANTHROPIC_API_KEY");
const ANTHROPIC_MODEL = "claude-haiku-4-5-20251001";

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

  let payload: { file_base64?: string; media_type?: string };
  try {
    payload = await req.json();
  } catch {
    return json({ error: "Invalid JSON body" }, 400);
  }

  const { file_base64, media_type } = payload;
  if (!file_base64 || !media_type) return json({ error: "file_base64 and media_type are required" }, 400);

  const isPdf = media_type === "application/pdf";
  const isImage = media_type.startsWith("image/");
  if (!isPdf && !isImage) return json({ error: "File must be a PDF or an image" }, 400);

  try {
    const resp = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: ANTHROPIC_MODEL,
        max_tokens: 2000,
        messages: [
          {
            role: "user",
            content: [
              isPdf
                ? { type: "document", source: { type: "base64", media_type, data: file_base64 } }
                : { type: "image", source: { type: "base64", media_type, data: file_base64 } },
              {
                type: "text",
                text:
                  "Transcribe this answer key exactly as printed -- a list of question numbers with their correct " +
                  "letter choice (e.g. '1. C  2. D  3. A ...'). Return ONLY that list as plain text, nothing else, " +
                  "no commentary.",
              },
            ],
          },
        ],
      }),
    });

    if (!resp.ok) {
      const errText = await resp.text();
      return json({ error: `Claude API error ${resp.status}: ${errText}` }, 502);
    }

    const data = await resp.json();
    const block = (data.content ?? []).find((b: { type: string }) => b.type === "text");
    const text = (block?.text ?? "").trim();
    if (!text) return json({ error: "Claude did not return any text for this file." }, 502);

    return json({ answer_key_text: text });
  } catch (err) {
    return json({ error: err instanceof Error ? err.message : String(err) }, 500);
  }
});

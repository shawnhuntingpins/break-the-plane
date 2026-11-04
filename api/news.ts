// Break the Plane -- live player news proxy.
//
// Pulls RotoWire's NFLNews XML feed (URL kept in the NEWS_FEED_URL environment
// variable, not in the repo) and flattens it to
//   { updated, count, players: { "<RotoWire player id>": { headline, notes, date, ts } } }
// keeping only the NEWEST item per player. The board matches on the RotoWire
// player id, which is the number at the end of each card's rotowireUrl.
// Cached for 5 minutes per function instance. Serves the last good copy if the
// feed errors, so a feed hiccup never blanks the news on the board.
export const config = { runtime: 'edge' };

let cache = { at: 0, body: null as string | null };

function unwrap(s: string): string {
  return s.replace(/^\s*<!\[CDATA\[/, "").replace(/\]\]>\s*$/, "")
    .replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"').replace(/&apos;/g, "'").trim();
}
function tag(block: string, name: string): string {
  const m = block.match(new RegExp("<" + name + "(?:\\s[^>]*)?>([\\s\\S]*?)</" + name + ">"));
  return m ? unwrap(m[1]) : "";
}
function shorten(s: string, max = 240): string {
  if (s.length <= max) return s;
  const cut = s.slice(0, max);
  const stop = Math.max(cut.lastIndexOf(". "), cut.lastIndexOf("? "), cut.lastIndexOf("! "));
  return stop > 80 ? cut.slice(0, stop + 1) : cut.replace(/\s+\S*$/, "") + "…";
}
const MONTHS = ["January", "February", "March", "April", "May", "June", "July",
  "August", "September", "October", "November", "December"];

export function parseNews(xml: string) {
  const players: Record<string, { headline: string; notes: string; date: string; ts: number }> = {};
  const blocks = xml.match(/<Update\b[\s\S]*?<\/Update>/g) || [];
  for (const b of blocks) {
    const id = (b.match(/<Player\s+Id="(\d+)"/) || [])[1];
    const iso = tag(b, "DateTime");
    const headline = tag(b, "Headline");
    const ts = Date.parse(iso);
    if (!id || !headline || isNaN(ts)) continue;
    if (players[id] && players[id].ts >= ts) continue;
    // date as the feed writes it (Pacific, per the offset in DateTime)
    const [y, mo, d] = iso.slice(0, 10).split("-").map(Number);
    players[id] = {
      headline,
      notes: shorten(tag(b, "Notes")),
      date: MONTHS[mo - 1] + " " + d + ", " + y,
      ts,
    };
  }
  return { count: blocks.length, players };
}

export default async function (req: Request): Promise<Response> {
  const headers = {
    "Access-Control-Allow-Origin": "*",
    "Content-Type": "application/json",
    "Cache-Control": "public, max-age=0, s-maxage=300",
  };
  if (cache.body && Date.now() - cache.at < 300000) {
    return new Response(cache.body, { headers });
  }
  const url = process.env.NEWS_FEED_URL;
  if (!url) {
    return new Response(JSON.stringify({ error: "NEWS_FEED_URL env var not set" }), { status: 500, headers });
  }
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error("feed HTTP " + r.status);
    const { count, players } = parseNews(await r.text());
    if (!count) throw new Error("feed had no <Update> items");
    const body = JSON.stringify({ updated: Date.now(), count, players });
    cache = { at: Date.now(), body };
    return new Response(body, { headers });
  } catch (e) {
    if (cache.body) return new Response(cache.body, { headers });
    return new Response(JSON.stringify({ error: String(e) }), { status: 502, headers });
  }
}

// VOLLEYBALL MANIA 팀 저장 서버 (Cloudflare Workers + KV)
//   GET  /teams          팀 목록
//   GET  /teams/:name    팀의 저장된 위치
//   PUT  /teams/:name    { password, data } — 없으면 새로 만들고, 있으면 비밀번호 확인 후 덮어씀
//   DELETE /teams/:name         { password } — 팀 삭제
//   POST   /teams/:name/rename  { password, newName } — 팀 이름 변경
// 경기 영상 분석 (analysis.html)
//   GET    /matches             경기 목록
//   GET    /matches/:name       경기 기록
//   PUT    /matches/:name       { password, data, create } — create:true 면 새로 만들고, 아니면 비밀번호 확인 후 덮어씀
//   DELETE /matches/:name       { password } — 경기 삭제
const IDS = ['base', 'left', 'right', 'quick', 'a', 'b', 'ba', 'time'];
const MAX_TEAMS = 300;
const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET,PUT,POST,DELETE,OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Max-Age': '86400',
};
const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store', ...CORS },
});
const fail = (status, error) => json({ error }, status);

function cleanName(raw) {
  const name = String(raw || '').normalize('NFC').trim().replace(/\s+/g, ' ');
  if (!name || [...name].length > 20 || /[\u0000-\u001f\u007f\/\\]/.test(name)) return null;
  return name;
}
const isPoint = p => Array.isArray(p) && p.length === 2 && p.every(v => typeof v === 'number' && isFinite(v) && Math.abs(v) <= 200);
function cleanData(d) {
  if (!d || typeof d !== 'object') return null;
  const out = {};
  for (const id of IDS) {
    const v = d[id];
    if (!v || !Array.isArray(v.pos) || v.pos.length !== 9 || !v.pos.every(isPoint)) return null;
    out[id] = { pos: v.pos };
    for (const f of ['set', 'atk', 'target', 'target2', 'target3', 'decoy']) {
      if (v[f] === undefined) continue;
      if (!isPoint(v[f])) return null;
      out[id][f] = v[f];
    }
  }
  return out;
}
const hex = buf => [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2, '0')).join('');
async function hashPassword(password, saltHex) {
  const salt = new Uint8Array(saltHex.match(/../g).map(h => parseInt(h, 16)));
  const key = await crypto.subtle.importKey('raw', new TextEncoder().encode(password), 'PBKDF2', false, ['deriveBits']);
  return hex(await crypto.subtle.deriveBits({ name: 'PBKDF2', hash: 'SHA-256', salt, iterations: 100000 }, key, 256));
}

/* ---------- 경기 영상 분석 ---------- */
const MAX_MATCHES = 500;
const str = (v, n) => typeof v === 'string' ? v.slice(0, n) : '';
const num = (v, lo, hi) => typeof v === 'number' && isFinite(v) && v >= lo && v <= hi;
const pt = p => Array.isArray(p) && p.length === 2 && num(p[0], -20, 120) && num(p[1], -20, 220);
function cleanMatch(d) {
  if (!d || typeof d !== 'object') return null;
  const team = t => {
    if (!t || typeof t !== 'object' || !Array.isArray(t.players) || t.players.length !== 9) return null;
    return { name: str(t.name, 20), players: t.players.map(p => ({ no: str(p && p.no, 3), name: str(p && p.name, 12) })) };
  };
  const home = team(d.home), away = team(d.away);
  if (!home || !away || !Array.isArray(d.rallies) || d.rallies.length > 3000) return null;
  const rallies = [];
  for (const r of d.rallies) {
    if (!r || !Array.isArray(r.pos) || r.pos.length !== 18 || !r.pos.every(pt) || (r.win !== 0 && r.win !== 1)) return null;
    const t = num(r.t, 0, 1e6) ? r.t : null;
    let acts;
    if (Array.isArray(r.acts)) {               // v2: 동작 단위 기록 (스카우팅 코드)
      if (r.acts.length > 80) return null;
      acts = r.acts.map(cleanAct);
      if (acts.some(a => !a)) return null;
    } else if (Array.isArray(r.atks)) {        // v1: 공격 화살표만 있던 예전 형식 → 공격 동작으로
      if (r.atks.length > 30) return null;
      acts = [];
      for (const a of r.atks) {
        if (!a || (a.side !== 0 && a.side !== 1) || !idx(a.p) || !pt(a.from) || !pt(a.to) || !(a.r in OLD_RESULT)) return null;
        acts.push({ side: a.side, p: a.p, sk: 'A', ev: OLD_RESULT[a.r], t, from: a.from, to: a.to });
      }
    } else return null;
    rallies.push({
      t, pos: r.pos, acts, win: r.win,
      why: ['atk', 'blk', 'srv', 'err'].includes(r.why) ? r.why : 'atk',
      by: idx(r.by) ? r.by : null,
    });
  }
  const out = { v: 2, home, away, video: str(d.video, 300), setPoints: [15, 21, 25].includes(d.setPoints) ? d.setPoints : 21, rallies };
  if (d.base != null) {                      // 보드 기본 위치(18명) — 없으면 생략
    if (!Array.isArray(d.base) || d.base.length !== 18 || !d.base.every(pt)) return null;
    out.base = d.base;
  }
  return out;
}
const idx = v => Number.isInteger(v) && v >= 0 && v <= 8;
const OLD_RESULT = { kill: '#', err: '=', blk: '/', cont: '-' };
// 동작: S 서브 · R 리시브 · E 토스 · A 공격 · B 블로킹 · D 디그 · F 프리볼 / 평가: # + ! - / =
function cleanAct(a) {
  if (!a || typeof a !== 'object' || (a.side !== 0 && a.side !== 1) || !idx(a.p)) return null;
  if (typeof a.sk !== 'string' || a.sk.length !== 1 || !'SREABDF'.includes(a.sk)) return null;
  if (typeof a.ev !== 'string' || a.ev.length !== 1 || !'#+!-/='.includes(a.ev)) return null;
  const o = { side: a.side, p: a.p, sk: a.sk, ev: a.ev, t: num(a.t, 0, 1e6) ? a.t : null };
  for (const f of ['from', 'to']) {
    if (a[f] == null) continue;
    if (!pt(a[f])) return null;
    o[f] = a[f];
  }
  return o;
}
function matchScore(m) {          // 세트 스코어 "2-1" (목록 표시용)
  const s = [0, 0]; let p = [0, 0];
  for (const r of m.rallies) {
    if (s[0] >= 2 || s[1] >= 2) break;
    p[r.win]++;
    const w = r.win, l = 1 - w;
    if (p[w] >= m.setPoints && p[w] - p[l] >= 2) { s[w]++; p = [0, 0]; }
  }
  return s.join('-');
}
async function matches(req, env, parts) {
  if (parts.length === 1) {
    if (req.method !== 'GET') return fail(405, 'method_not_allowed');
    const list = [];
    let cursor;
    do {
      const page = await env.TEAMS.list({ prefix: 'match:', cursor });
      for (const k of page.keys) list.push({ name: k.metadata?.name || k.name.slice(6), updatedAt: k.metadata?.updatedAt || null,
        home: k.metadata?.home || '', away: k.metadata?.away || '', score: k.metadata?.score || '' });
      cursor = page.list_complete ? null : page.cursor;
    } while (cursor);
    list.sort((a, b) => String(b.updatedAt).localeCompare(String(a.updatedAt)));
    return json({ matches: list });
  }
  if (parts.length !== 2) return fail(404, 'not_found');
  let name;
  try { name = String(decodeURIComponent(parts[1])).normalize('NFC').trim().replace(/\s+/g, ' '); } catch (e) { name = ''; }
  if (!name || [...name].length > 40 || /[\u0000-\u001f\u007f\/\\]/.test(name)) return fail(400, 'bad_name');
  const key = 'match:' + name;
  if (!['GET', 'PUT', 'DELETE'].includes(req.method)) return fail(405, 'method_not_allowed');
  const stored = await env.TEAMS.get(key, 'json');
  if (req.method === 'GET') {
    if (!stored) return fail(404, 'no_match');
    return json({ name: stored.name, data: stored.data, updatedAt: stored.updatedAt });
  }
  const text = await req.text();
  if (text.length > 1500000) return fail(413, 'too_large');
  let body;
  try { body = JSON.parse(text); } catch (e) { return fail(400, 'bad_json'); }
  const password = typeof body.password === 'string' ? body.password : '';
  if (password.length < 4 || password.length > 64) return fail(400, 'bad_password');
  if (req.method === 'DELETE') {
    if (!stored) return fail(404, 'no_match');
    if (await hashPassword(password, stored.salt) !== stored.hash) return fail(401, 'wrong_password');
    await env.TEAMS.delete(key);
    return json({ deleted: name });
  }
  const data = cleanMatch(body.data);
  if (!data) return fail(400, 'bad_data');
  let salt, hash;
  if (stored) {
    if (body.create === true) return fail(409, 'match_exists');
    if (await hashPassword(password, stored.salt) !== stored.hash) return fail(401, 'wrong_password');
    ({ salt, hash } = stored);
  } else {
    if (body.create !== true) return fail(404, 'no_match');
    const list = await env.TEAMS.list({ prefix: 'match:', limit: 1000 });
    if (list.keys.length >= MAX_MATCHES) return fail(403, 'too_many_matches');
    salt = hex(crypto.getRandomValues(new Uint8Array(16)));
    hash = await hashPassword(password, salt);
  }
  const updatedAt = new Date().toISOString();
  const meta = { name, updatedAt, home: data.home.name, away: data.away.name, score: matchScore(data) };
  await env.TEAMS.put(key, JSON.stringify({ name, salt, hash, data, updatedAt }), { metadata: meta });
  return json({ name, updatedAt }, stored ? 200 : 201);
}

export default {
  async fetch(req, env) {
    if (req.method === 'OPTIONS') return new Response(null, { status: 204, headers: CORS });
    const parts = new URL(req.url).pathname.split('/').filter(Boolean);
    if (parts[0] === 'matches') return matches(req, env, parts);
    const renaming = parts.length === 3 && parts[2] === 'rename';
    if (parts[0] !== 'teams' || (parts.length > 2 && !renaming)) return fail(404, 'not_found');

    if (parts.length === 1) {
      if (req.method !== 'GET') return fail(405, 'method_not_allowed');
      const teams = [];
      let cursor;
      do {
        const page = await env.TEAMS.list({ prefix: 'team:', cursor });
        for (const k of page.keys) teams.push({ name: k.metadata?.name || k.name.slice(5), updatedAt: k.metadata?.updatedAt || null });
        cursor = page.list_complete ? null : page.cursor;
      } while (cursor);
      teams.sort((a, b) => a.name.localeCompare(b.name, 'ko'));
      return json({ teams });
    }

    let name;
    try { name = cleanName(decodeURIComponent(parts[1])); } catch (e) { name = null; }
    if (!name) return fail(400, 'bad_name');
    const key = 'team:' + name;
    const stored = await env.TEAMS.get(key, 'json');

    if (renaming ? req.method !== 'POST' : !['GET', 'PUT', 'DELETE'].includes(req.method)) return fail(405, 'method_not_allowed');
    if (req.method === 'GET') {
      if (!stored) return fail(404, 'no_team');
      return json({ name: stored.name, data: stored.data, updatedAt: stored.updatedAt });
    }

    const text = await req.text();
    if (text.length > 20000) return fail(413, 'too_large');
    let body;
    try { body = JSON.parse(text); } catch (e) { return fail(400, 'bad_json'); }
    const password = typeof body.password === 'string' ? body.password : '';
    if (password.length < 4 || password.length > 64) return fail(400, 'bad_password');
    if (req.method !== 'PUT') {          // 삭제·이름 변경
      if (!stored) return fail(404, 'no_team');
      if (await hashPassword(password, stored.salt) !== stored.hash) return fail(401, 'wrong_password');
      if (req.method === 'DELETE') { await env.TEAMS.delete(key); return json({ deleted: name }); }
      const newName = cleanName(body.newName);
      if (!newName) return fail(400, 'bad_name');
      if (newName === name) return json({ name });
      if (await env.TEAMS.get('team:' + newName)) return fail(409, 'team_exists');
      const updatedAt = new Date().toISOString();
      await env.TEAMS.put('team:' + newName, JSON.stringify({ ...stored, name: newName, updatedAt }), { metadata: { name: newName, updatedAt } });
      await env.TEAMS.delete(key);
      return json({ name: newName, updatedAt });
    }
    const data = cleanData(body.data);
    if (!data) return fail(400, 'bad_data');

    let salt, hash;
    if (stored) {
      if (await hashPassword(password, stored.salt) !== stored.hash) return fail(401, 'wrong_password');
      ({ salt, hash } = stored);
    } else {
      if (body.create !== true) return fail(404, 'no_team');
      const list = await env.TEAMS.list({ prefix: 'team:', limit: 1000 });
      if (list.keys.length >= MAX_TEAMS) return fail(403, 'too_many_teams');
      salt = hex(crypto.getRandomValues(new Uint8Array(16)));
      hash = await hashPassword(password, salt);
    }
    if (stored && body.create === true) return fail(409, 'team_exists');
    const updatedAt = new Date().toISOString();
    await env.TEAMS.put(key, JSON.stringify({ name, salt, hash, data, updatedAt }), { metadata: { name, updatedAt } });
    return json({ name, updatedAt }, stored ? 200 : 201);
  },
};

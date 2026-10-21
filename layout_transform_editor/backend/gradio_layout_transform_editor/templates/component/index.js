import { i as pi, g as wa, o as no, n as vr, u as $e, s as io, r as Kn, m as Sr, a as g, b as i, t as vi, d as ao, q as Ea, c as Sa, e as gr, f as Sn, h as yn, j as so, T as oo, k as lo, l as xn, p as Or, v as gi, w as Nr, x as Ta, y as Ma, z as Aa, A as nn, E as Tn, B as Gr, C as Oa, D as Nt, F as zi, G as uo, H as Na, I as Ha, J as fo, K as Xi, L as co, M as ho, N as mi, O as Qt, P as Ia, Q as Gn, R as po, S as vo, U as go, V as mo, W as _o, X as Pa, Y as _i, Z as Wi, _ as qi, $ as bo, a0 as yo, a1 as xo, a2 as wo, a3 as Mn, a4 as Eo, a5 as Ca, a6 as An, a7 as So, a8 as To, a9 as Mo, aa as Ao, ab as Oo, ac as No, ad as Ho, ae as Io, af as Po, ag as Co, ah as bi, ai as Lo, aj as Bo, ak as kt, al as Qn, am as $n, an as Ro, ao as ko, ap as Kt, aq as Go, ar as Do, as as Fo, at as jo, au as Jr, av as La, aw as Uo, ax as q, ay as Vo, az as Vr, aA as Yi, aB as zo, aC as ft, aD as zr, aE as U, aF as te, aG as be, aH as Xo, aI as J, aJ as ye, aK as Ke, aL as Wo, aM as Zi } from "./render-CIwk_WWh.js";
function Ba(e) {
  throw new Error("https://svelte.dev/e/lifecycle_outside_component");
}
const qo = [];
function Yo(e, t = !1, r = !1) {
  return mn(e, /* @__PURE__ */ new Map(), "", qo, null, r);
}
function mn(e, t, r, n, s = null, o = !1) {
  if (typeof e == "object" && e !== null) {
    var f = t.get(e);
    if (f !== void 0) return f;
    if (e instanceof Map) return (
      /** @type {Snapshot<T>} */
      new Map(e)
    );
    if (e instanceof Set) return (
      /** @type {Snapshot<T>} */
      new Set(e)
    );
    if (pi(e)) {
      var c = (
        /** @type {Snapshot<any>} */
        Array(e.length)
      );
      t.set(e, c), s !== null && t.set(s, c);
      for (var d = 0; d < e.length; d += 1) {
        var h = e[d];
        d in e && (c[d] = mn(h, t, r, n, null, o));
      }
      return c;
    }
    if (wa(e) === no) {
      c = {}, t.set(e, c), s !== null && t.set(s, c);
      for (var m of Object.keys(e))
        c[m] = mn(
          // @ts-expect-error
          e[m],
          t,
          r,
          n,
          null,
          o
        );
      return c;
    }
    if (e instanceof Date)
      return (
        /** @type {Snapshot<T>} */
        structuredClone(e)
      );
    if (typeof /** @type {T & { toJSON?: any } } */
    e.toJSON == "function" && !o)
      return mn(
        /** @type {T & { toJSON(): any } } */
        e.toJSON(),
        t,
        r,
        n,
        // Associate the instance with the toJSON clone
        e
      );
  }
  if (e instanceof EventTarget)
    return (
      /** @type {Snapshot<T>} */
      e
    );
  try {
    return (
      /** @type {Snapshot<T>} */
      structuredClone(e)
    );
  } catch {
    return (
      /** @type {Snapshot<T>} */
      e
    );
  }
}
function yi(e, t, r) {
  if (e == null)
    return t(void 0), r && r(void 0), vr;
  const n = $e(
    () => e.subscribe(
      t,
      // @ts-expect-error
      r
    )
  );
  return n.unsubscribe ? () => n.unsubscribe() : n;
}
const Rr = [];
function Zo(e, t) {
  return {
    subscribe: an(e, t).subscribe
  };
}
function an(e, t = vr) {
  let r = null;
  const n = /* @__PURE__ */ new Set();
  function s(c) {
    if (io(e, c) && (e = c, r)) {
      const d = !Rr.length;
      for (const h of n)
        h[1](), Rr.push(h, e);
      if (d) {
        for (let h = 0; h < Rr.length; h += 2)
          Rr[h][0](Rr[h + 1]);
        Rr.length = 0;
      }
    }
  }
  function o(c) {
    s(c(
      /** @type {T} */
      e
    ));
  }
  function f(c, d = vr) {
    const h = [c, d];
    return n.add(h), n.size === 1 && (r = t(s, o) || vr), c(
      /** @type {T} */
      e
    ), () => {
      n.delete(h), n.size === 0 && r && (r(), r = null);
    };
  }
  return { set: s, update: o, subscribe: f };
}
function Xr(e, t, r) {
  const n = !Array.isArray(e), s = n ? [e] : e;
  if (!s.every(Boolean))
    throw new Error("derived() expects stores as input, got a falsy value");
  const o = t.length < 2;
  return Zo(r, (f, c) => {
    let d = !1;
    const h = [];
    let m = 0, T = vr;
    const w = () => {
      if (m)
        return;
      T();
      const R = t(n ? h[0] : h, f, c);
      o ? f(R) : T = typeof R == "function" ? R : vr;
    }, L = s.map(
      (R, P) => yi(
        R,
        (H) => {
          h[P] = H, m &= ~(1 << P), d && w();
        },
        () => {
          m |= 1 << P;
        }
      )
    );
    return d = !0, w(), function() {
      Kn(L), T(), d = !1;
    };
  });
}
function Jo(e) {
  let t;
  return yi(e, (r) => t = r)(), t;
}
let pn = !1, ei = /* @__PURE__ */ Symbol("unmounted");
function Ji(e, t, r) {
  const n = r[t] ??= {
    store: null,
    source: Sr(void 0),
    unsubscribe: vr
  };
  if (n.store !== e && !(ei in r))
    if (n.unsubscribe(), n.store = e ?? null, e == null)
      n.source.v = void 0, n.unsubscribe = vr;
    else {
      var s = !0;
      n.unsubscribe = yi(e, (o) => {
        s ? n.source.v = o : g(n.source, o);
      }), s = !1;
    }
  return e && ei in r ? Jo(e) : i(n.source);
}
function Ko() {
  const e = {};
  function t() {
    vi(() => {
      for (var r in e)
        e[r].unsubscribe();
      ao(e, ei, {
        enumerable: !1,
        value: !0
      });
    });
  }
  return [e, t];
}
function Qo(e) {
  var t = pn;
  try {
    return pn = !1, [e(), pn];
  } finally {
    pn = t;
  }
}
function $o(e, t) {
  if (t) {
    const r = document.body;
    e.autofocus = !0, Ea(() => {
      document.activeElement === r && e.focus();
    });
  }
}
const el = (
  // We gotta write it like this because after downleveling the pure comment may end up in the wrong location
  globalThis?.window?.trustedTypes && /* @__PURE__ */ globalThis.window.trustedTypes.createPolicy("svelte-trusted-html", {
    /** @param {string} html */
    createHTML: (e) => e
  })
);
function tl(e) {
  return (
    /** @type {string} */
    el?.createHTML(e) ?? e
  );
}
function Ra(e) {
  var t = Sa("template");
  return t.innerHTML = tl(e.replaceAll("<!>", "<!---->")), t.content;
}
function Dr(e, t) {
  var r = (
    /** @type {Effect} */
    Sn
  );
  r.nodes === null && (r.nodes = { start: e, end: t, a: null, t: null });
}
// @__NO_SIDE_EFFECTS__
function xe(e, t) {
  var r = (t & oo) !== 0, n = (t & lo) !== 0, s, o = !e.startsWith("<!>");
  return () => {
    s === void 0 && (s = Ra(o ? e : "<!>" + e), r || (s = /** @type {TemplateNode} */
    yn(s)));
    var f = (
      /** @type {TemplateNode} */
      n || so ? document.importNode(s, !0) : s.cloneNode(!0)
    );
    if (r) {
      var c = (
        /** @type {TemplateNode} */
        yn(f)
      ), d = (
        /** @type {TemplateNode} */
        f.lastChild
      );
      Dr(c, d);
    } else
      Dr(f, f);
    return f;
  };
}
// @__NO_SIDE_EFFECTS__
function rl(e, t, r = "svg") {
  var n = !e.startsWith("<!>"), s = `<${r}>${n ? e : "<!>" + e}</${r}>`, o;
  return () => {
    if (!o) {
      var f = (
        /** @type {DocumentFragment} */
        Ra(s)
      ), c = (
        /** @type {Element} */
        yn(f)
      );
      o = /** @type {Element} */
      yn(c);
    }
    var d = (
      /** @type {TemplateNode} */
      o.cloneNode(!0)
    );
    return Dr(d, d), d;
  };
}
// @__NO_SIDE_EFFECTS__
function ka(e, t) {
  return /* @__PURE__ */ rl(e, t, "svg");
}
function Jt(e = "") {
  {
    var t = gr(e + "");
    return Dr(t, t), t;
  }
}
function Tr() {
  var e = document.createDocumentFragment(), t = document.createComment(""), r = gr();
  return e.append(t, r), Dr(t, r), e;
}
function Z(e, t) {
  e !== null && e.before(
    /** @type {Node} */
    t
  );
}
class On {
  /** @type {TemplateNode} */
  anchor;
  /** @type {Map<Batch, Key>} */
  #t = /* @__PURE__ */ new Map();
  /**
   * Map of keys to effects that are currently rendered in the DOM.
   * These effects are visible and actively part of the document tree.
   * Example:
   * ```
   * {#if condition}
   * 	foo
   * {:else}
   * 	bar
   * {/if}
   * ```
   * Can result in the entries `true->Effect` and `false->Effect`
   * @type {Map<Key, Effect>}
   */
  #r = /* @__PURE__ */ new Map();
  /**
   * Similar to #onscreen with respect to the keys, but contains branches that are not yet
   * in the DOM, because their insertion is deferred.
   * @type {Map<Key, Branch>}
   */
  #e = /* @__PURE__ */ new Map();
  /**
   * Keys of effects that are currently outroing
   * @type {Set<Key>}
   */
  #n = /* @__PURE__ */ new Set();
  /**
   * Whether to pause (i.e. outro) on change, or destroy immediately.
   * This is necessary for `<svelte:element>`
   */
  #i = !0;
  /**
   * @param {TemplateNode} anchor
   * @param {boolean} transition
   */
  constructor(t, r = !0) {
    this.anchor = t, this.#i = r;
  }
  /**
   * @param {Batch} batch
   */
  #a = (t) => {
    if (this.#t.has(t)) {
      var r = (
        /** @type {Key} */
        this.#t.get(t)
      ), n = this.#r.get(r);
      if (n)
        xn(n), this.#n.delete(r);
      else {
        var s = this.#e.get(r);
        s && (xn(s.effect), this.#r.set(r, s.effect), this.#e.delete(r), s.fragment.lastChild.remove(), this.anchor.before(s.fragment), n = s.effect);
      }
      for (const [o, f] of this.#t) {
        if (this.#t.delete(o), o === t)
          break;
        const c = this.#e.get(f);
        c && (Or(c.effect), this.#e.delete(f));
      }
      for (const [o, f] of this.#r) {
        if (o === r || this.#n.has(o)) continue;
        const c = () => {
          if (Array.from(this.#t.values()).includes(o)) {
            var h = document.createDocumentFragment();
            Ma(f, h), h.append(gr()), this.#e.set(o, { effect: f, fragment: h });
          } else
            Or(f);
          this.#n.delete(o), this.#r.delete(o);
        };
        this.#i || !n ? (this.#n.add(o), gi(f, c, !1)) : c();
      }
    }
  };
  /**
   * @param {Batch} batch
   */
  #s = (t) => {
    this.#t.delete(t);
    const r = Array.from(this.#t.values());
    for (const [n, s] of this.#e)
      r.includes(n) || (Or(s.effect), this.#e.delete(n));
  };
  /**
   *
   * @param {any} key
   * @param {null | ((target: TemplateNode) => void)} fn
   */
  ensure(t, r) {
    var n = (
      /** @type {Batch} */
      Ta
    ), s = Aa();
    if (r && !this.#r.has(t) && !this.#e.has(t))
      if (s) {
        var o = document.createDocumentFragment(), f = gr();
        o.append(f), this.#e.set(t, {
          effect: Nr(() => r(f)),
          fragment: o
        });
      } else
        this.#r.set(
          t,
          Nr(() => r(this.anchor))
        );
    if (this.#t.set(n, t), s) {
      for (const [c, d] of this.#r)
        c === t ? n.unskip_effect(d) : n.skip_effect(d);
      for (const [c, d] of this.#e)
        c === t ? n.unskip_effect(d.effect) : n.skip_effect(d.effect);
      n.oncommit(this.#a), n.ondiscard(this.#s);
    } else
      this.#a(n);
  }
}
function nl(e, t, ...r) {
  var n = new On(e);
  nn(() => {
    const s = t() ?? null;
    n.ensure(s, s && ((o) => s(o, ...r)));
  }, Tn);
}
function il(e) {
  Gr === null && Ba(), Oa && Gr.l !== null ? al(Gr).m.push(e) : Nt(() => {
    const t = $e(e);
    if (typeof t == "function") return (
      /** @type {() => void} */
      t
    );
  });
}
function Ga(e) {
  Gr === null && Ba(), il(() => () => $e(e));
}
function al(e) {
  var t = (
    /** @type {ComponentContextLegacy} */
    e.l
  );
  return t.u ??= { a: [], b: [], m: [] };
}
function _e(e, t, r = !1) {
  var n = new On(e), s = r ? Tn : 0;
  function o(f, c) {
    n.ensure(f, c);
  }
  nn(() => {
    var f = !1;
    t((c, d = 0) => {
      f = !0, o(d, c);
    }), f || o(-1, null);
  }, s);
}
function Mr(e, t) {
  return t;
}
function sl(e, t, r) {
  for (var n = [], s = t.length, o, f = t.length, c = 0; c < s; c++) {
    let T = t[c];
    gi(
      T,
      () => {
        if (o) {
          if (o.pending.delete(T), o.done.add(T), o.pending.size === 0) {
            var w = (
              /** @type {Set<EachOutroGroup>} */
              e.outrogroups
            );
            ti(e, mi(o.done)), w.delete(o), w.size === 0 && (e.outrogroups = null);
          }
        } else
          f -= 1;
      },
      !1
    );
  }
  if (f === 0) {
    var d = n.length === 0 && r !== null;
    if (d) {
      var h = (
        /** @type {Element} */
        r
      ), m = (
        /** @type {Element} */
        h.parentNode
      );
      go(m), m.append(h), e.items.clear();
    }
    ti(e, t, !d);
  } else
    o = {
      pending: new Set(t),
      done: /* @__PURE__ */ new Set()
    }, (e.outrogroups ??= /* @__PURE__ */ new Set()).add(o);
}
function ti(e, t, r = !0) {
  var n;
  if (e.pending.size > 0) {
    n = /* @__PURE__ */ new Set();
    for (const f of e.pending.values())
      for (const c of f)
        n.add(
          /** @type {EachItem} */
          e.items.get(c).e
        );
  }
  for (var s = 0; s < t.length; s++) {
    var o = t[s];
    if (n?.has(o)) {
      o.f |= Qt;
      const f = document.createDocumentFragment();
      Ma(o, f);
    } else
      Or(t[s], r);
  }
}
var Ki;
function Ar(e, t, r, n, s, o = null) {
  var f = e, c = /* @__PURE__ */ new Map(), d = (t & Na) !== 0;
  if (d) {
    var h = (
      /** @type {Element} */
      e
    );
    f = h.appendChild(gr());
  }
  var m = null, T = Ha(() => {
    var S = r();
    return (
      /** @type {V[]} */
      pi(S) ? S : S == null ? [] : mi(S)
    );
  }), w, L = /* @__PURE__ */ new Map(), R = !0;
  function P(S) {
    (y.effect.f & Ia) === 0 && (y.pending.delete(S), y.fallback = m, ol(y, w, f, t, n), m !== null && (w.length === 0 ? (m.f & Qt) === 0 ? xn(m) : (m.f ^= Qt, $r(m, null, f)) : gi(m, () => {
      m = null;
    })));
  }
  function H(S) {
    y.pending.delete(S);
  }
  var _ = nn(() => {
    w = /** @type {V[]} */
    i(T);
    for (var S = w.length, E = /* @__PURE__ */ new Set(), b = (
      /** @type {Batch} */
      Ta
    ), A = Aa(), N = 0; N < S; N += 1) {
      var W = w[N], Y = n(W, N), V = R ? null : c.get(Y);
      V ? (V.v && zi(V.v, W), V.i && zi(V.i, N), A && b.unskip_effect(V.e)) : (V = ll(
        c,
        R ? f : Ki ??= gr(),
        W,
        Y,
        N,
        s,
        t,
        r
      ), R || (V.e.f |= Qt), c.set(Y, V)), E.add(Y);
    }
    if (S === 0 && o && !m && (R ? m = Nr(() => o(f)) : (m = Nr(() => o(Ki ??= gr())), m.f |= Qt)), S > E.size && uo(), !R)
      if (L.set(b, E), A) {
        for (const [Ce, Ge] of c)
          E.has(Ce) || b.skip_effect(Ge.e);
        b.oncommit(P), b.ondiscard(H);
      } else
        P(b);
    i(T);
  }), y = { effect: _, items: c, pending: L, outrogroups: null, fallback: m };
  R = !1;
}
function Kr(e) {
  for (; e !== null && (e.f & po) === 0; )
    e = e.next;
  return e;
}
function ol(e, t, r, n, s) {
  var o = (n & vo) !== 0, f = t.length, c = e.items, d = Kr(e.effect.first), h, m = null, T, w = [], L = [], R, P, H, _;
  if (o)
    for (_ = 0; _ < f; _ += 1)
      R = t[_], P = s(R, _), H = /** @type {EachItem} */
      c.get(P).e, (H.f & Qt) === 0 && (H.nodes?.a?.measure(), (T ??= /* @__PURE__ */ new Set()).add(H));
  for (_ = 0; _ < f; _ += 1) {
    if (R = t[_], P = s(R, _), H = /** @type {EachItem} */
    c.get(P).e, e.outrogroups !== null)
      for (const V of e.outrogroups)
        V.pending.delete(H), V.done.delete(H);
    if ((H.f & Gn) !== 0 && (xn(H), o && (H.nodes?.a?.unfix(), (T ??= /* @__PURE__ */ new Set()).delete(H))), (H.f & Qt) !== 0)
      if (H.f ^= Qt, H === d)
        $r(H, null, r);
      else {
        var y = m ? m.next : d;
        H === e.effect.last && (e.effect.last = H.prev), H.prev && (H.prev.next = H.next), H.next && (H.next.prev = H.prev), dr(e, m, H), dr(e, H, y), $r(H, y, r), m = H, w = [], L = [], d = Kr(m.next);
        continue;
      }
    if (H !== d) {
      if (h !== void 0 && h.has(H)) {
        if (w.length < L.length) {
          var S = L[0], E;
          m = S.prev;
          var b = w[0], A = w[w.length - 1];
          for (E = 0; E < w.length; E += 1)
            $r(w[E], S, r);
          for (E = 0; E < L.length; E += 1)
            h.delete(L[E]);
          dr(e, b.prev, A.next), dr(e, m, b), dr(e, A, S), d = S, m = A, _ -= 1, w = [], L = [];
        } else
          h.delete(H), $r(H, d, r), dr(e, H.prev, H.next), dr(e, H, m === null ? e.effect.first : m.next), dr(e, m, H), m = H;
        continue;
      }
      for (w = [], L = []; d !== null && d !== H; )
        (h ??= /* @__PURE__ */ new Set()).add(d), L.push(d), d = Kr(d.next);
      if (d === null)
        continue;
    }
    (H.f & Qt) === 0 && w.push(H), m = H, d = Kr(H.next);
  }
  if (e.outrogroups !== null) {
    for (const V of e.outrogroups)
      V.pending.size === 0 && (ti(e, mi(V.done)), e.outrogroups?.delete(V));
    e.outrogroups.size === 0 && (e.outrogroups = null);
  }
  if (d !== null || h !== void 0) {
    var N = [];
    if (h !== void 0)
      for (H of h)
        (H.f & Gn) === 0 && N.push(H);
    for (; d !== null; )
      (d.f & Gn) === 0 && d !== e.fallback && N.push(d), d = Kr(d.next);
    var W = N.length;
    if (W > 0) {
      var Y = (n & Na) !== 0 && f === 0 ? r : null;
      if (o) {
        for (_ = 0; _ < W; _ += 1)
          N[_].nodes?.a?.measure();
        for (_ = 0; _ < W; _ += 1)
          N[_].nodes?.a?.fix();
      }
      sl(e, N, Y);
    }
  }
  o && Ea(() => {
    if (T !== void 0)
      for (H of T)
        H.nodes?.a?.apply();
  });
}
function ll(e, t, r, n, s, o, f, c) {
  var d = (f & co) !== 0 ? (f & ho) === 0 ? Sr(r, !1, !1) : Xi(r) : null, h = (f & fo) !== 0 ? Xi(s) : null;
  return {
    v: d,
    i: h,
    e: Nr(() => (o(t, d ?? r, h ?? s, c), () => {
      e.delete(n);
    }))
  };
}
function $r(e, t, r) {
  if (e.nodes)
    for (var n = e.nodes.start, s = e.nodes.end, o = t && (t.f & Qt) === 0 ? (
      /** @type {EffectNodes} */
      t.nodes.start
    ) : r; n !== null; ) {
      var f = (
        /** @type {TemplateNode} */
        mo(n)
      );
      if (o.before(n), n === s)
        return;
      n = f;
    }
}
function dr(e, t, r) {
  t === null ? e.effect.first = r : t.next = r, r === null ? e.effect.last = t : r.prev = t;
}
function ri(e, t, r, n, s) {
  var o = t.$$slots?.[r], f = !1;
  o === !0 && (o = t[r === "default" ? "children" : r], f = !0), o === void 0 || o(e, f ? () => n : n);
}
function ul(e, t, r) {
  var n = new On(e);
  nn(() => {
    var s = t() ?? null;
    n.ensure(s, s && ((o) => r(o, s)));
  }, Tn);
}
const fl = () => performance.now(), Wt = {
  // don't access requestAnimationFrame eagerly outside method
  // this allows basic testing of user code without JSDOM
  // bunder will eval and remove ternary when the user's app is built
  tick: (
    /** @param {any} _ */
    (e) => requestAnimationFrame(e)
  ),
  now: () => fl(),
  tasks: /* @__PURE__ */ new Set()
};
function Da() {
  const e = Wt.now();
  Wt.tasks.forEach((t) => {
    t.c(e) || (Wt.tasks.delete(t), t.f());
  }), Wt.tasks.size !== 0 && Wt.tick(Da);
}
function cl(e) {
  let t;
  return Wt.tasks.size === 0 && Wt.tick(Da), {
    promise: new Promise((r) => {
      Wt.tasks.add(t = { c: e, f: r });
    }),
    abort() {
      Wt.tasks.delete(t);
    }
  };
}
function hl(e, t, r, n, s, o) {
  var f = null, c = (
    /** @type {TemplateNode} */
    e
  ), d = new On(c, !1);
  nn(() => {
    const h = t() || null;
    var m = h === "svg" ? _o : void 0;
    if (h === null) {
      d.ensure(null, null);
      return;
    }
    return d.ensure(h, (T) => {
      if (h) {
        if (f = Sa(h, m), Dr(f, f), n) {
          var w = null, L = f.appendChild(gr());
          n(f, L), w?.remove();
        }
        Sn.nodes.end = f, T.before(f);
      }
    }), () => {
    };
  }, Tn), vi(() => {
  });
}
function dl(e, t) {
  var r = void 0, n;
  Pa(() => {
    r !== (r = t()) && (n && (Or(n), n = null), r && (n = Nr(() => {
      _i(() => (
        /** @type {(node: Element) => void} */
        r(e)
      ));
    })));
  });
}
function Fa(e) {
  var t, r, n = "";
  if (typeof e == "string" || typeof e == "number") n += e;
  else if (typeof e == "object") if (Array.isArray(e)) {
    var s = e.length;
    for (t = 0; t < s; t++) e[t] && (r = Fa(e[t])) && (n && (n += " "), n += r);
  } else for (r in e) e[r] && (n && (n += " "), n += r);
  return n;
}
function pl() {
  for (var e, t, r = 0, n = "", s = arguments.length; r < s; r++) (e = arguments[r]) && (t = Fa(e)) && (n && (n += " "), n += t);
  return n;
}
function vl(e) {
  return typeof e == "object" ? pl(e) : e ?? "";
}
const Qi = [...` 	
\r\f \v\uFEFF`];
function gl(e, t, r) {
  var n = e == null ? "" : "" + e;
  if (t && (n = n ? n + " " + t : t), r) {
    for (var s of Object.keys(r))
      if (r[s])
        n = n ? n + " " + s : s;
      else if (n.length)
        for (var o = s.length, f = 0; (f = n.indexOf(s, f)) >= 0; ) {
          var c = f + o;
          (f === 0 || Qi.includes(n[f - 1])) && (c === n.length || Qi.includes(n[c])) ? n = (f === 0 ? "" : n.substring(0, f)) + n.substring(c + 1) : f = c;
        }
  }
  return n === "" ? null : n;
}
function $i(e, t = !1) {
  var r = t ? " !important;" : ";", n = "";
  for (var s of Object.keys(e)) {
    var o = e[s];
    o != null && o !== "" && (n += " " + s + ": " + o + r);
  }
  return n;
}
function Dn(e) {
  return e[0] !== "-" || e[1] !== "-" ? e.toLowerCase() : e;
}
function ml(e, t) {
  if (t) {
    var r = "", n, s;
    if (Array.isArray(t) ? (n = t[0], s = t[1]) : n = t, e) {
      e = String(e).replaceAll(/\s*\/\*.*?\*\/\s*/g, "").trim();
      var o = !1, f = 0, c = !1, d = [];
      n && d.push(...Object.keys(n).map(Dn)), s && d.push(...Object.keys(s).map(Dn));
      var h = 0, m = -1;
      const P = e.length;
      for (var T = 0; T < P; T++) {
        var w = e[T];
        if (c ? w === "/" && e[T - 1] === "*" && (c = !1) : o ? o === w && (o = !1) : w === "/" && e[T + 1] === "*" ? c = !0 : w === '"' || w === "'" ? o = w : w === "(" ? f++ : w === ")" && f--, !c && o === !1 && f === 0) {
          if (w === ":" && m === -1)
            m = T;
          else if (w === ";" || T === P - 1) {
            if (m !== -1) {
              var L = Dn(e.substring(h, m).trim());
              if (!d.includes(L)) {
                w !== ";" && T++;
                var R = e.substring(h, T).trim();
                r += " " + R + ";";
              }
            }
            h = T + 1, m = -1;
          }
        }
      }
    }
    return n && (r += $i(n)), s && (r += $i(s, !0)), r = r.trim(), r === "" ? null : r;
  }
  return e == null ? null : String(e);
}
function Ot(e, t, r, n, s, o) {
  var f = (
    /** @type {any} */
    e[Wi]
  );
  if (f !== r || f === void 0) {
    var c = gl(r, n, o);
    c == null ? e.removeAttribute("class") : t ? e.className = c : e.setAttribute("class", c), e[Wi] = r;
  } else if (o && s !== o)
    for (var d in o) {
      var h = !!o[d];
      (s == null || h !== !!s[d]) && e.classList.toggle(d, h);
    }
  return o;
}
function Fn(e, t = {}, r, n) {
  for (var s in r) {
    var o = r[s];
    t[s] !== o && (r[s] == null ? e.style.removeProperty(s) : e.style.setProperty(s, o, n));
  }
}
function Ht(e, t, r, n) {
  var s = (
    /** @type {any} */
    e[qi]
  );
  if (s !== t) {
    var o = ml(t, n);
    o == null ? e.removeAttribute("style") : e.style.cssText = o, e[qi] = t;
  } else n && (Array.isArray(n) ? (Fn(e, r?.[0], n[0]), Fn(e, r?.[1], n[1], "important")) : Fn(e, r, n));
  return n;
}
function tn(e, t, r = !1) {
  if (e.multiple) {
    if (t == null)
      return;
    if (!pi(t))
      return bo();
    for (var n of e.options)
      n.selected = t.includes(ea(n));
    return;
  }
  for (n of e.options) {
    var s = ea(n);
    if (yo(s, t)) {
      n.selected = !0;
      return;
    }
  }
  (!r || t !== void 0) && (e.selectedIndex = -1);
}
function xi(e) {
  var t = new MutationObserver(() => {
    tn(e, e.__value);
  });
  t.observe(e, {
    // Listen to option element changes
    childList: !0,
    subtree: !0,
    // because of <optgroup>
    // Listen to option element value attribute changes
    // (doesn't get notified of select value changes,
    // because that property is not reflected as an attribute)
    attributes: !0,
    attributeFilter: ["value"]
  }), vi(() => {
    t.disconnect();
  });
}
function ea(e) {
  return "__value" in e ? e.__value : e.value;
}
const en = /* @__PURE__ */ Symbol("class"), kr = /* @__PURE__ */ Symbol("style"), ja = /* @__PURE__ */ Symbol("is custom element"), Ua = /* @__PURE__ */ Symbol("is html"), _l = Mn ? "input" : "INPUT", bl = Mn ? "option" : "OPTION", yl = Mn ? "select" : "SELECT", xl = Mn ? "progress" : "PROGRESS";
function wl(e, t) {
  var r = wi(e);
  r.value === (r.value = // treat null and undefined the same for the initial value
  t ?? void 0) || // @ts-expect-error
  // `progress` elements always need their value set when it's `0`
  e.value === t && (t !== 0 || e.nodeName !== xl) || (e.value = t ?? "");
}
function El(e, t) {
  t ? e.hasAttribute("selected") || e.setAttribute("selected", "") : e.removeAttribute("selected");
}
function mr(e, t, r, n) {
  var s = wi(e);
  s[t] !== (s[t] = r) && (t === "loading" && (e[No] = r), r == null ? e.removeAttribute(t) : typeof r != "string" && Va(e).includes(t) ? e[t] = r : e.setAttribute(t, r));
}
function Sl(e, t, r, n, s = !1, o = !1) {
  var f = wi(e), c = f[ja], d = !f[Ua], h = t || {}, m = e.nodeName === bl;
  for (var T in t)
    T in r || (r[T] = null);
  r.class ? r.class = vl(r.class) : r.class = null, r[kr] && (r.style ??= null);
  var w = Va(e);
  if (e.nodeName === _l && "type" in r && ("value" in r || "__value" in r)) {
    var L = r.type;
    (L !== h.type || L === void 0 && e.hasAttribute("type")) && (h.type = L, mr(e, "type", L));
  }
  for (const E in r) {
    let b = r[E];
    if (m && E === "value" && b == null) {
      e.value = e.__value = "", h[E] = b;
      continue;
    }
    if (E === "class") {
      var R = e.namespaceURI === "http://www.w3.org/1999/xhtml";
      Ot(e, R, b, n, t?.[en], r[en]), h[E] = b, h[en] = r[en];
      continue;
    }
    if (E === "style") {
      Ht(e, b, t?.[kr], r[kr]), h[E] = b, h[kr] = r[kr];
      continue;
    }
    var P = h[E];
    if (!(b === P && !(b === void 0 && e.hasAttribute(E)))) {
      h[E] = b;
      var H = E[0] + E[1];
      if (H !== "$$")
        if (H === "on") {
          const A = {}, N = "$$" + E;
          let W = E.slice(2);
          var _ = Ho(W);
          if (Eo(W) && (W = W.slice(0, -7), A.capture = !0), !_ && P) {
            if (b != null) continue;
            e.removeEventListener(W, h[N], A), h[N] = null;
          }
          if (_)
            Ca(W, e, b), An([W]);
          else if (b != null) {
            let Y = function(V) {
              h[E].call(this, V);
            };
            h[N] = So(W, e, Y, A);
          }
        } else if (E === "style")
          mr(e, E, b);
        else if (E === "autofocus")
          $o(
            /** @type {HTMLElement} */
            e,
            !!b
          );
        else if (!c && (E === "__value" || E === "value" && b != null))
          e.value = e.__value = b;
        else if (E === "selected" && m)
          El(
            /** @type {HTMLOptionElement} */
            e,
            b
          );
        else {
          var y = E;
          d || (y = To(y));
          var S = y === "defaultValue" || y === "defaultChecked";
          if (b == null && !c && !S)
            if (f[E] = null, y === "value" || y === "checked") {
              let A = (
                /** @type {HTMLInputElement} */
                e
              );
              const N = t === void 0;
              if (y === "value") {
                let W = A.defaultValue;
                A.removeAttribute(y), A.defaultValue = W, A.value = A.__value = N ? W : null;
              } else {
                let W = A.defaultChecked;
                A.removeAttribute(y), A.defaultChecked = W, A.checked = N ? W : !1;
              }
            } else
              e.removeAttribute(E);
          else S || w.includes(y) && (c || typeof b != "string") ? (e[y] = b, y in f && (f[y] = Mo)) : typeof b != "function" && mr(e, y, b);
        }
    }
  }
  return h;
}
function Tl(e, t, r = [], n = [], s = [], o, f = !1, c = !1) {
  xo(s, r, n, (d) => {
    var h = void 0, m = {}, T = e.nodeName === yl, w = !1;
    if (Pa(() => {
      var R = t(...d.map(i)), P = Sl(
        e,
        h,
        R,
        o,
        f,
        c
      );
      w && T && "value" in R && tn(
        /** @type {HTMLSelectElement} */
        e,
        R.value
      );
      for (let _ of Object.getOwnPropertySymbols(m))
        R[_] || Or(m[_]);
      for (let _ of Object.getOwnPropertySymbols(R)) {
        var H = R[_];
        _.description === wo && (!h || H !== h[_]) && (m[_] && Or(m[_]), m[_] = Nr(() => dl(e, () => H))), P[_] = H;
      }
      h = P;
    }), T) {
      var L = (
        /** @type {HTMLSelectElement} */
        e
      );
      _i(() => {
        tn(
          L,
          /** @type {Record<string | symbol, any>} */
          h.value,
          !0
        ), xi(L);
      });
    }
    w = !0;
  });
}
function wi(e) {
  return (
    /** @type {Record<string | symbol, unknown>} **/
    /** @type {any} */
    e[Ao] ??= {
      [ja]: e.nodeName.includes("-"),
      [Ua]: e.namespaceURI === Oo
    }
  );
}
var ta = /* @__PURE__ */ new Map();
function Va(e) {
  var t = e.getAttribute("is") || e.nodeName, r = ta.get(t);
  if (r) return r;
  ta.set(t, r = []);
  for (var n, s = e, o = Element.prototype; o !== s; ) {
    n = Io(s);
    for (var f in n)
      n[f].set && // better safe than sorry, we don't want spread attributes to mess with HTML content
      f !== "innerHTML" && f !== "textContent" && f !== "innerText" && r.push(f);
    s = wa(s);
  }
  return r;
}
function jn(e, t) {
  return e === t || e?.[bi] === t;
}
function Nn(e = {}, t, r, n) {
  var s = (
    /** @type {ComponentContext} */
    Gr.r
  ), o = (
    /** @type {Effect} */
    Sn
  );
  return _i(() => {
    var f, c;
    return Po(() => {
      f = c, c = [], $e(() => {
        jn(r(...c), e) || (t(e, ...c), f && jn(r(...f), e) && t(null, ...f));
      });
    }), () => {
      let d = o;
      for (; d !== s && d.parent !== null && d.parent.f & Co; )
        d = d.parent;
      const h = () => {
        c && jn(r(...c), e) && t(null, ...c);
      }, m = d.teardown;
      d.teardown = () => {
        h(), m?.();
      };
    };
  }), e;
}
function Ml(e = !1) {
  const t = (
    /** @type {ComponentContextLegacy} */
    Gr
  ), r = t.l.u;
  if (!r) return;
  let n = () => kt(t.s);
  if (e) {
    let s = 0, o = (
      /** @type {Record<string, any>} */
      {}
    );
    const f = Qn(() => {
      let c = !1;
      const d = t.s;
      for (const h in d)
        d[h] !== o[h] && (o[h] = d[h], c = !0);
      return c && s++, s;
    });
    n = () => i(f);
  }
  r.b.length && Lo(() => {
    ra(t, n), Kn(r.b);
  }), Nt(() => {
    const s = $e(() => r.m.map(Bo));
    return () => {
      for (const o of s)
        typeof o == "function" && o();
    };
  }), r.a.length && Nt(() => {
    ra(t, n), Kn(r.a);
  });
}
function ra(e, t) {
  if (e.l.s)
    for (const r of e.l.s) i(r);
  t();
}
const Al = {
  get(e, t) {
    if (!e.exclude.has(t))
      return e.props[t];
  },
  set(e, t) {
    return !1;
  },
  getOwnPropertyDescriptor(e, t) {
    if (!e.exclude.has(t) && t in e.props)
      return {
        enumerable: !0,
        configurable: !0,
        value: e.props[t]
      };
  },
  has(e, t) {
    return e.exclude.has(t) ? !1 : t in e.props;
  },
  ownKeys(e) {
    return Reflect.ownKeys(e.props).filter((t) => !e.exclude.has(t));
  }
};
// @__NO_SIDE_EFFECTS__
function Ei(e, t, r) {
  return new Proxy(
    { props: e, exclude: t },
    Al
  );
}
const Ol = {
  get(e, t) {
    let r = e.props.length;
    for (; r--; ) {
      let n = e.props[r];
      if (Jr(n) && (n = n()), typeof n == "object" && n !== null && t in n) return n[t];
    }
  },
  set(e, t, r) {
    let n = e.props.length;
    for (; n--; ) {
      let s = e.props[n];
      Jr(s) && (s = s());
      const o = $n(s, t);
      if (o && o.set)
        return o.set(r), !0;
    }
    return !1;
  },
  getOwnPropertyDescriptor(e, t) {
    let r = e.props.length;
    for (; r--; ) {
      let n = e.props[r];
      if (Jr(n) && (n = n()), typeof n == "object" && n !== null && t in n) {
        const s = $n(n, t);
        return s && !s.configurable && (s.configurable = !0), s;
      }
    }
  },
  has(e, t) {
    if (t === bi || t === La) return !1;
    for (let r of e.props)
      if (Jr(r) && (r = r()), r != null && t in r) return !0;
    return !1;
  },
  ownKeys(e) {
    const t = [];
    for (let r of e.props)
      if (Jr(r) && (r = r()), !!r) {
        for (const n in r)
          t.includes(n) || t.push(n);
        for (const n of Object.getOwnPropertySymbols(r))
          t.includes(n) || t.push(n);
      }
    return t;
  }
};
function ni(...e) {
  return new Proxy({ props: e }, Ol);
}
function X(e, t, r, n) {
  var s = !Oa || (r & Fo) !== 0, o = (r & Do) !== 0, f = (r & Uo) !== 0, c = (
    /** @type {V} */
    n
  ), d = !0, h = (
    /** @type {Derived<V> | undefined} */
    void 0
  ), m = () => f && s ? (h ??= Qn(
    /** @type {() => V} */
    n
  ), i(h)) : (d && (d = !1, c = f ? $e(
    /** @type {() => V} */
    n
  ) : (
    /** @type {V} */
    n
  )), c);
  let T;
  if (o) {
    var w = bi in e || La in e;
    T = $n(e, t)?.set ?? (w && t in e ? (E) => e[t] = E : void 0);
  }
  var L, R = !1;
  o ? [L, R] = Qo(() => (
    /** @type {V} */
    e[t]
  )) : L = /** @type {V} */
  e[t], L === void 0 && n !== void 0 && (L = m(), T && (s && Ro(), T(L)));
  var P;
  if (s ? P = () => {
    var E = (
      /** @type {V} */
      e[t]
    );
    return E === void 0 ? m() : (d = !0, E);
  } : P = () => {
    var E = (
      /** @type {V} */
      e[t]
    );
    return E !== void 0 && (c = /** @type {V} */
    void 0), E === void 0 ? c : E;
  }, s && (r & ko) === 0)
    return P;
  if (T) {
    var H = e.$$legacy;
    return (
      /** @type {() => V} */
      (function(E, b) {
        return arguments.length > 0 ? ((!s || !b || H || R) && T(b ? P() : E), E) : P();
      })
    );
  }
  var _ = !1, y = ((r & jo) !== 0 ? Qn : Ha)(() => (_ = !1, P()));
  o && i(y);
  var S = (
    /** @type {Effect} */
    Sn
  );
  return (
    /** @type {() => V} */
    (function(E, b) {
      if (arguments.length > 0) {
        const A = b ? i(y) : s && o ? Kt(E) : E;
        return g(y, A), _ = !0, c !== void 0 && (c = A), E;
      }
      return Go && _ || (S.f & Ia) !== 0 ? y.v : i(y);
    })
  );
}
const Nl = [
  { color: "red", primary: 600, secondary: 100 },
  { color: "green", primary: 600, secondary: 100 },
  { color: "blue", primary: 600, secondary: 100 },
  { color: "yellow", primary: 500, secondary: 100 },
  { color: "purple", primary: 600, secondary: 100 },
  { color: "teal", primary: 600, secondary: 100 },
  { color: "orange", primary: 600, secondary: 100 },
  { color: "cyan", primary: 600, secondary: 100 },
  { color: "lime", primary: 500, secondary: 100 },
  { color: "pink", primary: 600, secondary: 100 }
], na = {
  inherit: "inherit",
  current: "currentColor",
  transparent: "transparent",
  black: "#000",
  white: "#fff",
  slate: {
    50: "#f8fafc",
    100: "#f1f5f9",
    200: "#e2e8f0",
    300: "#cbd5e1",
    400: "#94a3b8",
    500: "#64748b",
    600: "#475569",
    700: "#334155",
    800: "#1e293b",
    900: "#0f172a",
    950: "#020617"
  },
  gray: {
    50: "#f9fafb",
    100: "#f3f4f6",
    200: "#e5e7eb",
    300: "#d1d5db",
    400: "#9ca3af",
    500: "#6b7280",
    600: "#4b5563",
    700: "#374151",
    800: "#1f2937",
    900: "#111827",
    950: "#030712"
  },
  zinc: {
    50: "#fafafa",
    100: "#f4f4f5",
    200: "#e4e4e7",
    300: "#d4d4d8",
    400: "#a1a1aa",
    500: "#71717a",
    600: "#52525b",
    700: "#3f3f46",
    800: "#27272a",
    900: "#18181b",
    950: "#09090b"
  },
  neutral: {
    50: "#fafafa",
    100: "#f5f5f5",
    200: "#e5e5e5",
    300: "#d4d4d4",
    400: "#a3a3a3",
    500: "#737373",
    600: "#525252",
    700: "#404040",
    800: "#262626",
    900: "#171717",
    950: "#0a0a0a"
  },
  stone: {
    50: "#fafaf9",
    100: "#f5f5f4",
    200: "#e7e5e4",
    300: "#d6d3d1",
    400: "#a8a29e",
    500: "#78716c",
    600: "#57534e",
    700: "#44403c",
    800: "#292524",
    900: "#1c1917",
    950: "#0c0a09"
  },
  red: {
    50: "#fef2f2",
    100: "#fee2e2",
    200: "#fecaca",
    300: "#fca5a5",
    400: "#f87171",
    500: "#ef4444",
    600: "#dc2626",
    700: "#b91c1c",
    800: "#991b1b",
    900: "#7f1d1d",
    950: "#450a0a"
  },
  orange: {
    50: "#fff7ed",
    100: "#ffedd5",
    200: "#fed7aa",
    300: "#fdba74",
    400: "#fb923c",
    500: "#f97316",
    600: "#ea580c",
    700: "#c2410c",
    800: "#9a3412",
    900: "#7c2d12",
    950: "#431407"
  },
  amber: {
    50: "#fffbeb",
    100: "#fef3c7",
    200: "#fde68a",
    300: "#fcd34d",
    400: "#fbbf24",
    500: "#f59e0b",
    600: "#d97706",
    700: "#b45309",
    800: "#92400e",
    900: "#78350f",
    950: "#451a03"
  },
  yellow: {
    50: "#fefce8",
    100: "#fef9c3",
    200: "#fef08a",
    300: "#fde047",
    400: "#facc15",
    500: "#eab308",
    600: "#ca8a04",
    700: "#a16207",
    800: "#854d0e",
    900: "#713f12",
    950: "#422006"
  },
  lime: {
    50: "#f7fee7",
    100: "#ecfccb",
    200: "#d9f99d",
    300: "#bef264",
    400: "#a3e635",
    500: "#84cc16",
    600: "#65a30d",
    700: "#4d7c0f",
    800: "#3f6212",
    900: "#365314",
    950: "#1a2e05"
  },
  green: {
    50: "#f0fdf4",
    100: "#dcfce7",
    200: "#bbf7d0",
    300: "#86efac",
    400: "#4ade80",
    500: "#22c55e",
    600: "#16a34a",
    700: "#15803d",
    800: "#166534",
    900: "#14532d",
    950: "#052e16"
  },
  emerald: {
    50: "#ecfdf5",
    100: "#d1fae5",
    200: "#a7f3d0",
    300: "#6ee7b7",
    400: "#34d399",
    500: "#10b981",
    600: "#059669",
    700: "#047857",
    800: "#065f46",
    900: "#064e3b",
    950: "#022c22"
  },
  teal: {
    50: "#f0fdfa",
    100: "#ccfbf1",
    200: "#99f6e4",
    300: "#5eead4",
    400: "#2dd4bf",
    500: "#14b8a6",
    600: "#0d9488",
    700: "#0f766e",
    800: "#115e59",
    900: "#134e4a",
    950: "#042f2e"
  },
  cyan: {
    50: "#ecfeff",
    100: "#cffafe",
    200: "#a5f3fc",
    300: "#67e8f9",
    400: "#22d3ee",
    500: "#06b6d4",
    600: "#0891b2",
    700: "#0e7490",
    800: "#155e75",
    900: "#164e63",
    950: "#083344"
  },
  sky: {
    50: "#f0f9ff",
    100: "#e0f2fe",
    200: "#bae6fd",
    300: "#7dd3fc",
    400: "#38bdf8",
    500: "#0ea5e9",
    600: "#0284c7",
    700: "#0369a1",
    800: "#075985",
    900: "#0c4a6e",
    950: "#082f49"
  },
  blue: {
    50: "#eff6ff",
    100: "#dbeafe",
    200: "#bfdbfe",
    300: "#93c5fd",
    400: "#60a5fa",
    500: "#3b82f6",
    600: "#2563eb",
    700: "#1d4ed8",
    800: "#1e40af",
    900: "#1e3a8a",
    950: "#172554"
  },
  indigo: {
    50: "#eef2ff",
    100: "#e0e7ff",
    200: "#c7d2fe",
    300: "#a5b4fc",
    400: "#818cf8",
    500: "#6366f1",
    600: "#4f46e5",
    700: "#4338ca",
    800: "#3730a3",
    900: "#312e81",
    950: "#1e1b4b"
  },
  violet: {
    50: "#f5f3ff",
    100: "#ede9fe",
    200: "#ddd6fe",
    300: "#c4b5fd",
    400: "#a78bfa",
    500: "#8b5cf6",
    600: "#7c3aed",
    700: "#6d28d9",
    800: "#5b21b6",
    900: "#4c1d95",
    950: "#2e1065"
  },
  purple: {
    50: "#faf5ff",
    100: "#f3e8ff",
    200: "#e9d5ff",
    300: "#d8b4fe",
    400: "#c084fc",
    500: "#a855f7",
    600: "#9333ea",
    700: "#7e22ce",
    800: "#6b21a8",
    900: "#581c87",
    950: "#3b0764"
  },
  fuchsia: {
    50: "#fdf4ff",
    100: "#fae8ff",
    200: "#f5d0fe",
    300: "#f0abfc",
    400: "#e879f9",
    500: "#d946ef",
    600: "#c026d3",
    700: "#a21caf",
    800: "#86198f",
    900: "#701a75",
    950: "#4a044e"
  },
  pink: {
    50: "#fdf2f8",
    100: "#fce7f3",
    200: "#fbcfe8",
    300: "#f9a8d4",
    400: "#f472b6",
    500: "#ec4899",
    600: "#db2777",
    700: "#be185d",
    800: "#9d174d",
    900: "#831843",
    950: "#500724"
  },
  rose: {
    50: "#fff1f2",
    100: "#ffe4e6",
    200: "#fecdd3",
    300: "#fda4af",
    400: "#fb7185",
    500: "#f43f5e",
    600: "#e11d48",
    700: "#be123c",
    800: "#9f1239",
    900: "#881337",
    950: "#4c0519"
  }
};
Nl.reduce((e, { color: t, primary: r, secondary: n }) => ({
  ...e,
  [t]: {
    primary: na[t][r],
    secondary: na[t][n]
  }
}), {});
function Hl(e) {
  return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, "default") ? e.default : e;
}
var Un, ia;
function Il() {
  if (ia) return Un;
  ia = 1;
  var e = function(y) {
    return t(y) && !r(y);
  };
  function t(_) {
    return !!_ && typeof _ == "object";
  }
  function r(_) {
    var y = Object.prototype.toString.call(_);
    return y === "[object RegExp]" || y === "[object Date]" || o(_);
  }
  var n = typeof Symbol == "function" && Symbol.for, s = n ? /* @__PURE__ */ Symbol.for("react.element") : 60103;
  function o(_) {
    return _.$$typeof === s;
  }
  function f(_) {
    return Array.isArray(_) ? [] : {};
  }
  function c(_, y) {
    return y.clone !== !1 && y.isMergeableObject(_) ? P(f(_), _, y) : _;
  }
  function d(_, y, S) {
    return _.concat(y).map(function(E) {
      return c(E, S);
    });
  }
  function h(_, y) {
    if (!y.customMerge)
      return P;
    var S = y.customMerge(_);
    return typeof S == "function" ? S : P;
  }
  function m(_) {
    return Object.getOwnPropertySymbols ? Object.getOwnPropertySymbols(_).filter(function(y) {
      return Object.propertyIsEnumerable.call(_, y);
    }) : [];
  }
  function T(_) {
    return Object.keys(_).concat(m(_));
  }
  function w(_, y) {
    try {
      return y in _;
    } catch {
      return !1;
    }
  }
  function L(_, y) {
    return w(_, y) && !(Object.hasOwnProperty.call(_, y) && Object.propertyIsEnumerable.call(_, y));
  }
  function R(_, y, S) {
    var E = {};
    return S.isMergeableObject(_) && T(_).forEach(function(b) {
      E[b] = c(_[b], S);
    }), T(y).forEach(function(b) {
      L(_, b) || (w(_, b) && S.isMergeableObject(y[b]) ? E[b] = h(b, S)(_[b], y[b], S) : E[b] = c(y[b], S));
    }), E;
  }
  function P(_, y, S) {
    S = S || {}, S.arrayMerge = S.arrayMerge || d, S.isMergeableObject = S.isMergeableObject || e, S.cloneUnlessOtherwiseSpecified = c;
    var E = Array.isArray(y), b = Array.isArray(_), A = E === b;
    return A ? E ? S.arrayMerge(_, y, S) : R(_, y, S) : c(y, S);
  }
  P.all = function(y, S) {
    if (!Array.isArray(y))
      throw new Error("first argument should be an array");
    return y.reduce(function(E, b) {
      return P(E, b, S);
    }, {});
  };
  var H = P;
  return Un = H, Un;
}
var Pl = Il();
const Cl = /* @__PURE__ */ Hl(Pl);
var ii = function(e, t) {
  return ii = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(r, n) {
    r.__proto__ = n;
  } || function(r, n) {
    for (var s in n) Object.prototype.hasOwnProperty.call(n, s) && (r[s] = n[s]);
  }, ii(e, t);
};
function Hn(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Class extends value " + String(t) + " is not a constructor or null");
  ii(e, t);
  function r() {
    this.constructor = e;
  }
  e.prototype = t === null ? Object.create(t) : (r.prototype = t.prototype, new r());
}
var he = function() {
  return he = Object.assign || function(t) {
    for (var r, n = 1, s = arguments.length; n < s; n++) {
      r = arguments[n];
      for (var o in r) Object.prototype.hasOwnProperty.call(r, o) && (t[o] = r[o]);
    }
    return t;
  }, he.apply(this, arguments);
};
function Ll(e, t) {
  var r = {};
  for (var n in e) Object.prototype.hasOwnProperty.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && typeof Object.getOwnPropertySymbols == "function")
    for (var s = 0, n = Object.getOwnPropertySymbols(e); s < n.length; s++)
      t.indexOf(n[s]) < 0 && Object.prototype.propertyIsEnumerable.call(e, n[s]) && (r[n[s]] = e[n[s]]);
  return r;
}
function Vn(e, t, r) {
  if (r || arguments.length === 2) for (var n = 0, s = t.length, o; n < s; n++)
    (o || !(n in t)) && (o || (o = Array.prototype.slice.call(t, 0, n)), o[n] = t[n]);
  return e.concat(o || Array.prototype.slice.call(t));
}
function zn(e, t) {
  var r = t && t.cache ? t.cache : jl, n = t && t.serializer ? t.serializer : Dl, s = t && t.strategy ? t.strategy : kl;
  return s(e, {
    cache: r,
    serializer: n
  });
}
function Bl(e) {
  return e == null || typeof e == "number" || typeof e == "boolean";
}
function Rl(e, t, r, n) {
  var s = Bl(n) ? n : r(n), o = t.get(s);
  return typeof o > "u" && (o = e.call(this, n), t.set(s, o)), o;
}
function za(e, t, r) {
  var n = Array.prototype.slice.call(arguments, 3), s = r(n), o = t.get(s);
  return typeof o > "u" && (o = e.apply(this, n), t.set(s, o)), o;
}
function Xa(e, t, r, n, s) {
  return r.bind(t, e, n, s);
}
function kl(e, t) {
  var r = e.length === 1 ? Rl : za;
  return Xa(e, this, r, t.cache.create(), t.serializer);
}
function Gl(e, t) {
  return Xa(e, this, za, t.cache.create(), t.serializer);
}
var Dl = function() {
  return JSON.stringify(arguments);
}, Fl = (
  /** @class */
  (function() {
    function e() {
      this.cache = /* @__PURE__ */ Object.create(null);
    }
    return e.prototype.get = function(t) {
      return this.cache[t];
    }, e.prototype.set = function(t, r) {
      this.cache[t] = r;
    }, e;
  })()
), jl = {
  create: function() {
    return new Fl();
  }
}, Xn = {
  variadic: Gl
}, le;
(function(e) {
  e[e.EXPECT_ARGUMENT_CLOSING_BRACE = 1] = "EXPECT_ARGUMENT_CLOSING_BRACE", e[e.EMPTY_ARGUMENT = 2] = "EMPTY_ARGUMENT", e[e.MALFORMED_ARGUMENT = 3] = "MALFORMED_ARGUMENT", e[e.EXPECT_ARGUMENT_TYPE = 4] = "EXPECT_ARGUMENT_TYPE", e[e.INVALID_ARGUMENT_TYPE = 5] = "INVALID_ARGUMENT_TYPE", e[e.EXPECT_ARGUMENT_STYLE = 6] = "EXPECT_ARGUMENT_STYLE", e[e.INVALID_NUMBER_SKELETON = 7] = "INVALID_NUMBER_SKELETON", e[e.INVALID_DATE_TIME_SKELETON = 8] = "INVALID_DATE_TIME_SKELETON", e[e.EXPECT_NUMBER_SKELETON = 9] = "EXPECT_NUMBER_SKELETON", e[e.EXPECT_DATE_TIME_SKELETON = 10] = "EXPECT_DATE_TIME_SKELETON", e[e.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE = 11] = "UNCLOSED_QUOTE_IN_ARGUMENT_STYLE", e[e.EXPECT_SELECT_ARGUMENT_OPTIONS = 12] = "EXPECT_SELECT_ARGUMENT_OPTIONS", e[e.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE = 13] = "EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE = 14] = "INVALID_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR = 15] = "EXPECT_SELECT_ARGUMENT_SELECTOR", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR = 16] = "EXPECT_PLURAL_ARGUMENT_SELECTOR", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT = 17] = "EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT = 18] = "EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT", e[e.INVALID_PLURAL_ARGUMENT_SELECTOR = 19] = "INVALID_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_PLURAL_ARGUMENT_SELECTOR = 20] = "DUPLICATE_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_SELECT_ARGUMENT_SELECTOR = 21] = "DUPLICATE_SELECT_ARGUMENT_SELECTOR", e[e.MISSING_OTHER_CLAUSE = 22] = "MISSING_OTHER_CLAUSE", e[e.INVALID_TAG = 23] = "INVALID_TAG", e[e.INVALID_TAG_NAME = 25] = "INVALID_TAG_NAME", e[e.UNMATCHED_CLOSING_TAG = 26] = "UNMATCHED_CLOSING_TAG", e[e.UNCLOSED_TAG = 27] = "UNCLOSED_TAG";
})(le || (le = {}));
var Re;
(function(e) {
  e[e.literal = 0] = "literal", e[e.argument = 1] = "argument", e[e.number = 2] = "number", e[e.date = 3] = "date", e[e.time = 4] = "time", e[e.select = 5] = "select", e[e.plural = 6] = "plural", e[e.pound = 7] = "pound", e[e.tag = 8] = "tag";
})(Re || (Re = {}));
var Fr;
(function(e) {
  e[e.number = 0] = "number", e[e.dateTime = 1] = "dateTime";
})(Fr || (Fr = {}));
function aa(e) {
  return e.type === Re.literal;
}
function Ul(e) {
  return e.type === Re.argument;
}
function Wa(e) {
  return e.type === Re.number;
}
function qa(e) {
  return e.type === Re.date;
}
function Ya(e) {
  return e.type === Re.time;
}
function Za(e) {
  return e.type === Re.select;
}
function Ja(e) {
  return e.type === Re.plural;
}
function Vl(e) {
  return e.type === Re.pound;
}
function Ka(e) {
  return e.type === Re.tag;
}
function Qa(e) {
  return !!(e && typeof e == "object" && e.type === Fr.number);
}
function ai(e) {
  return !!(e && typeof e == "object" && e.type === Fr.dateTime);
}
var $a = /[ \xA0\u1680\u2000-\u200A\u202F\u205F\u3000]/, zl = /(?:[Eec]{1,6}|G{1,5}|[Qq]{1,5}|(?:[yYur]+|U{1,5})|[ML]{1,5}|d{1,2}|D{1,3}|F{1}|[abB]{1,5}|[hkHK]{1,2}|w{1,2}|W{1}|m{1,2}|s{1,2}|[zZOvVxX]{1,4})(?=([^']*'[^']*')*[^']*$)/g;
function Xl(e) {
  var t = {};
  return e.replace(zl, function(r) {
    var n = r.length;
    switch (r[0]) {
      // Era
      case "G":
        t.era = n === 4 ? "long" : n === 5 ? "narrow" : "short";
        break;
      // Year
      case "y":
        t.year = n === 2 ? "2-digit" : "numeric";
        break;
      case "Y":
      case "u":
      case "U":
      case "r":
        throw new RangeError("`Y/u/U/r` (year) patterns are not supported, use `y` instead");
      // Quarter
      case "q":
      case "Q":
        throw new RangeError("`q/Q` (quarter) patterns are not supported");
      // Month
      case "M":
      case "L":
        t.month = ["numeric", "2-digit", "short", "long", "narrow"][n - 1];
        break;
      // Week
      case "w":
      case "W":
        throw new RangeError("`w/W` (week) patterns are not supported");
      case "d":
        t.day = ["numeric", "2-digit"][n - 1];
        break;
      case "D":
      case "F":
      case "g":
        throw new RangeError("`D/F/g` (day) patterns are not supported, use `d` instead");
      // Weekday
      case "E":
        t.weekday = n === 4 ? "long" : n === 5 ? "narrow" : "short";
        break;
      case "e":
        if (n < 4)
          throw new RangeError("`e..eee` (weekday) patterns are not supported");
        t.weekday = ["short", "long", "narrow", "short"][n - 4];
        break;
      case "c":
        if (n < 4)
          throw new RangeError("`c..ccc` (weekday) patterns are not supported");
        t.weekday = ["short", "long", "narrow", "short"][n - 4];
        break;
      // Period
      case "a":
        t.hour12 = !0;
        break;
      case "b":
      // am, pm, noon, midnight
      case "B":
        throw new RangeError("`b/B` (period) patterns are not supported, use `a` instead");
      // Hour
      case "h":
        t.hourCycle = "h12", t.hour = ["numeric", "2-digit"][n - 1];
        break;
      case "H":
        t.hourCycle = "h23", t.hour = ["numeric", "2-digit"][n - 1];
        break;
      case "K":
        t.hourCycle = "h11", t.hour = ["numeric", "2-digit"][n - 1];
        break;
      case "k":
        t.hourCycle = "h24", t.hour = ["numeric", "2-digit"][n - 1];
        break;
      case "j":
      case "J":
      case "C":
        throw new RangeError("`j/J/C` (hour) patterns are not supported, use `h/H/K/k` instead");
      // Minute
      case "m":
        t.minute = ["numeric", "2-digit"][n - 1];
        break;
      // Second
      case "s":
        t.second = ["numeric", "2-digit"][n - 1];
        break;
      case "S":
      case "A":
        throw new RangeError("`S/A` (second) patterns are not supported, use `s` instead");
      // Zone
      case "z":
        t.timeZoneName = n < 4 ? "short" : "long";
        break;
      case "Z":
      // 1..3, 4, 5: The ISO8601 varios formats
      case "O":
      // 1, 4: milliseconds in day short, long
      case "v":
      // 1, 4: generic non-location format
      case "V":
      // 1, 2, 3, 4: time zone ID or city
      case "X":
      // 1, 2, 3, 4: The ISO8601 varios formats
      case "x":
        throw new RangeError("`Z/O/v/V/X/x` (timeZone) patterns are not supported, use `z` instead");
    }
    return "";
  }), t;
}
var Wl = /[\t-\r \x85\u200E\u200F\u2028\u2029]/i;
function ql(e) {
  if (e.length === 0)
    throw new Error("Number skeleton cannot be empty");
  for (var t = e.split(Wl).filter(function(w) {
    return w.length > 0;
  }), r = [], n = 0, s = t; n < s.length; n++) {
    var o = s[n], f = o.split("/");
    if (f.length === 0)
      throw new Error("Invalid number skeleton");
    for (var c = f[0], d = f.slice(1), h = 0, m = d; h < m.length; h++) {
      var T = m[h];
      if (T.length === 0)
        throw new Error("Invalid number skeleton");
    }
    r.push({ stem: c, options: d });
  }
  return r;
}
function Yl(e) {
  return e.replace(/^(.*?)-/, "");
}
var sa = /^\.(?:(0+)(\*)?|(#+)|(0+)(#+))$/g, es = /^(@+)?(\+|#+)?[rs]?$/g, Zl = /(\*)(0+)|(#+)(0+)|(0+)/g, ts = /^(0+)$/;
function oa(e) {
  var t = {};
  return e[e.length - 1] === "r" ? t.roundingPriority = "morePrecision" : e[e.length - 1] === "s" && (t.roundingPriority = "lessPrecision"), e.replace(es, function(r, n, s) {
    return typeof s != "string" ? (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length) : s === "+" ? t.minimumSignificantDigits = n.length : n[0] === "#" ? t.maximumSignificantDigits = n.length : (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length + (typeof s == "string" ? s.length : 0)), "";
  }), t;
}
function rs(e) {
  switch (e) {
    case "sign-auto":
      return {
        signDisplay: "auto"
      };
    case "sign-accounting":
    case "()":
      return {
        currencySign: "accounting"
      };
    case "sign-always":
    case "+!":
      return {
        signDisplay: "always"
      };
    case "sign-accounting-always":
    case "()!":
      return {
        signDisplay: "always",
        currencySign: "accounting"
      };
    case "sign-except-zero":
    case "+?":
      return {
        signDisplay: "exceptZero"
      };
    case "sign-accounting-except-zero":
    case "()?":
      return {
        signDisplay: "exceptZero",
        currencySign: "accounting"
      };
    case "sign-never":
    case "+_":
      return {
        signDisplay: "never"
      };
  }
}
function Jl(e) {
  var t;
  if (e[0] === "E" && e[1] === "E" ? (t = {
    notation: "engineering"
  }, e = e.slice(2)) : e[0] === "E" && (t = {
    notation: "scientific"
  }, e = e.slice(1)), t) {
    var r = e.slice(0, 2);
    if (r === "+!" ? (t.signDisplay = "always", e = e.slice(2)) : r === "+?" && (t.signDisplay = "exceptZero", e = e.slice(2)), !ts.test(e))
      throw new Error("Malformed concise eng/scientific notation");
    t.minimumIntegerDigits = e.length;
  }
  return t;
}
function la(e) {
  var t = {}, r = rs(e);
  return r || t;
}
function Kl(e) {
  for (var t = {}, r = 0, n = e; r < n.length; r++) {
    var s = n[r];
    switch (s.stem) {
      case "percent":
      case "%":
        t.style = "percent";
        continue;
      case "%x100":
        t.style = "percent", t.scale = 100;
        continue;
      case "currency":
        t.style = "currency", t.currency = s.options[0];
        continue;
      case "group-off":
      case ",_":
        t.useGrouping = !1;
        continue;
      case "precision-integer":
      case ".":
        t.maximumFractionDigits = 0;
        continue;
      case "measure-unit":
      case "unit":
        t.style = "unit", t.unit = Yl(s.options[0]);
        continue;
      case "compact-short":
      case "K":
        t.notation = "compact", t.compactDisplay = "short";
        continue;
      case "compact-long":
      case "KK":
        t.notation = "compact", t.compactDisplay = "long";
        continue;
      case "scientific":
        t = he(he(he({}, t), { notation: "scientific" }), s.options.reduce(function(d, h) {
          return he(he({}, d), la(h));
        }, {}));
        continue;
      case "engineering":
        t = he(he(he({}, t), { notation: "engineering" }), s.options.reduce(function(d, h) {
          return he(he({}, d), la(h));
        }, {}));
        continue;
      case "notation-simple":
        t.notation = "standard";
        continue;
      // https://github.com/unicode-org/icu/blob/master/icu4c/source/i18n/unicode/unumberformatter.h
      case "unit-width-narrow":
        t.currencyDisplay = "narrowSymbol", t.unitDisplay = "narrow";
        continue;
      case "unit-width-short":
        t.currencyDisplay = "code", t.unitDisplay = "short";
        continue;
      case "unit-width-full-name":
        t.currencyDisplay = "name", t.unitDisplay = "long";
        continue;
      case "unit-width-iso-code":
        t.currencyDisplay = "symbol";
        continue;
      case "scale":
        t.scale = parseFloat(s.options[0]);
        continue;
      case "rounding-mode-floor":
        t.roundingMode = "floor";
        continue;
      case "rounding-mode-ceiling":
        t.roundingMode = "ceil";
        continue;
      case "rounding-mode-down":
        t.roundingMode = "trunc";
        continue;
      case "rounding-mode-up":
        t.roundingMode = "expand";
        continue;
      case "rounding-mode-half-even":
        t.roundingMode = "halfEven";
        continue;
      case "rounding-mode-half-down":
        t.roundingMode = "halfTrunc";
        continue;
      case "rounding-mode-half-up":
        t.roundingMode = "halfExpand";
        continue;
      // https://unicode-org.github.io/icu/userguide/format_parse/numbers/skeletons.html#integer-width
      case "integer-width":
        if (s.options.length > 1)
          throw new RangeError("integer-width stems only accept a single optional option");
        s.options[0].replace(Zl, function(d, h, m, T, w, L) {
          if (h)
            t.minimumIntegerDigits = m.length;
          else {
            if (T && w)
              throw new Error("We currently do not support maximum integer digits");
            if (L)
              throw new Error("We currently do not support exact integer digits");
          }
          return "";
        });
        continue;
    }
    if (ts.test(s.stem)) {
      t.minimumIntegerDigits = s.stem.length;
      continue;
    }
    if (sa.test(s.stem)) {
      if (s.options.length > 1)
        throw new RangeError("Fraction-precision stems only accept a single optional option");
      s.stem.replace(sa, function(d, h, m, T, w, L) {
        return m === "*" ? t.minimumFractionDigits = h.length : T && T[0] === "#" ? t.maximumFractionDigits = T.length : w && L ? (t.minimumFractionDigits = w.length, t.maximumFractionDigits = w.length + L.length) : (t.minimumFractionDigits = h.length, t.maximumFractionDigits = h.length), "";
      });
      var o = s.options[0];
      o === "w" ? t = he(he({}, t), { trailingZeroDisplay: "stripIfInteger" }) : o && (t = he(he({}, t), oa(o)));
      continue;
    }
    if (es.test(s.stem)) {
      t = he(he({}, t), oa(s.stem));
      continue;
    }
    var f = rs(s.stem);
    f && (t = he(he({}, t), f));
    var c = Jl(s.stem);
    c && (t = he(he({}, t), c));
  }
  return t;
}
var vn = {
  "001": [
    "H",
    "h"
  ],
  419: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  AC: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  AD: [
    "H",
    "hB"
  ],
  AE: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  AF: [
    "H",
    "hb",
    "hB",
    "h"
  ],
  AG: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  AI: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  AL: [
    "h",
    "H",
    "hB"
  ],
  AM: [
    "H",
    "hB"
  ],
  AO: [
    "H",
    "hB"
  ],
  AR: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  AS: [
    "h",
    "H"
  ],
  AT: [
    "H",
    "hB"
  ],
  AU: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  AW: [
    "H",
    "hB"
  ],
  AX: [
    "H"
  ],
  AZ: [
    "H",
    "hB",
    "h"
  ],
  BA: [
    "H",
    "hB",
    "h"
  ],
  BB: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  BD: [
    "h",
    "hB",
    "H"
  ],
  BE: [
    "H",
    "hB"
  ],
  BF: [
    "H",
    "hB"
  ],
  BG: [
    "H",
    "hB",
    "h"
  ],
  BH: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  BI: [
    "H",
    "h"
  ],
  BJ: [
    "H",
    "hB"
  ],
  BL: [
    "H",
    "hB"
  ],
  BM: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  BN: [
    "hb",
    "hB",
    "h",
    "H"
  ],
  BO: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  BQ: [
    "H"
  ],
  BR: [
    "H",
    "hB"
  ],
  BS: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  BT: [
    "h",
    "H"
  ],
  BW: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  BY: [
    "H",
    "h"
  ],
  BZ: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  CA: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  CC: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  CD: [
    "hB",
    "H"
  ],
  CF: [
    "H",
    "h",
    "hB"
  ],
  CG: [
    "H",
    "hB"
  ],
  CH: [
    "H",
    "hB",
    "h"
  ],
  CI: [
    "H",
    "hB"
  ],
  CK: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  CL: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  CM: [
    "H",
    "h",
    "hB"
  ],
  CN: [
    "H",
    "hB",
    "hb",
    "h"
  ],
  CO: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  CP: [
    "H"
  ],
  CR: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  CU: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  CV: [
    "H",
    "hB"
  ],
  CW: [
    "H",
    "hB"
  ],
  CX: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  CY: [
    "h",
    "H",
    "hb",
    "hB"
  ],
  CZ: [
    "H"
  ],
  DE: [
    "H",
    "hB"
  ],
  DG: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  DJ: [
    "h",
    "H"
  ],
  DK: [
    "H"
  ],
  DM: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  DO: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  DZ: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  EA: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  EC: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  EE: [
    "H",
    "hB"
  ],
  EG: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  EH: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  ER: [
    "h",
    "H"
  ],
  ES: [
    "H",
    "hB",
    "h",
    "hb"
  ],
  ET: [
    "hB",
    "hb",
    "h",
    "H"
  ],
  FI: [
    "H"
  ],
  FJ: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  FK: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  FM: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  FO: [
    "H",
    "h"
  ],
  FR: [
    "H",
    "hB"
  ],
  GA: [
    "H",
    "hB"
  ],
  GB: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  GD: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  GE: [
    "H",
    "hB",
    "h"
  ],
  GF: [
    "H",
    "hB"
  ],
  GG: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  GH: [
    "h",
    "H"
  ],
  GI: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  GL: [
    "H",
    "h"
  ],
  GM: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  GN: [
    "H",
    "hB"
  ],
  GP: [
    "H",
    "hB"
  ],
  GQ: [
    "H",
    "hB",
    "h",
    "hb"
  ],
  GR: [
    "h",
    "H",
    "hb",
    "hB"
  ],
  GT: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  GU: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  GW: [
    "H",
    "hB"
  ],
  GY: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  HK: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  HN: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  HR: [
    "H",
    "hB"
  ],
  HU: [
    "H",
    "h"
  ],
  IC: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  ID: [
    "H"
  ],
  IE: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  IL: [
    "H",
    "hB"
  ],
  IM: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  IN: [
    "h",
    "H"
  ],
  IO: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  IQ: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  IR: [
    "hB",
    "H"
  ],
  IS: [
    "H"
  ],
  IT: [
    "H",
    "hB"
  ],
  JE: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  JM: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  JO: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  JP: [
    "H",
    "K",
    "h"
  ],
  KE: [
    "hB",
    "hb",
    "H",
    "h"
  ],
  KG: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  KH: [
    "hB",
    "h",
    "H",
    "hb"
  ],
  KI: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  KM: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  KN: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  KP: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  KR: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  KW: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  KY: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  KZ: [
    "H",
    "hB"
  ],
  LA: [
    "H",
    "hb",
    "hB",
    "h"
  ],
  LB: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  LC: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  LI: [
    "H",
    "hB",
    "h"
  ],
  LK: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  LR: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  LS: [
    "h",
    "H"
  ],
  LT: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  LU: [
    "H",
    "h",
    "hB"
  ],
  LV: [
    "H",
    "hB",
    "hb",
    "h"
  ],
  LY: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  MA: [
    "H",
    "h",
    "hB",
    "hb"
  ],
  MC: [
    "H",
    "hB"
  ],
  MD: [
    "H",
    "hB"
  ],
  ME: [
    "H",
    "hB",
    "h"
  ],
  MF: [
    "H",
    "hB"
  ],
  MG: [
    "H",
    "h"
  ],
  MH: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  MK: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  ML: [
    "H"
  ],
  MM: [
    "hB",
    "hb",
    "H",
    "h"
  ],
  MN: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  MO: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  MP: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  MQ: [
    "H",
    "hB"
  ],
  MR: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  MS: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  MT: [
    "H",
    "h"
  ],
  MU: [
    "H",
    "h"
  ],
  MV: [
    "H",
    "h"
  ],
  MW: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  MX: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  MY: [
    "hb",
    "hB",
    "h",
    "H"
  ],
  MZ: [
    "H",
    "hB"
  ],
  NA: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  NC: [
    "H",
    "hB"
  ],
  NE: [
    "H"
  ],
  NF: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  NG: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  NI: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  NL: [
    "H",
    "hB"
  ],
  NO: [
    "H",
    "h"
  ],
  NP: [
    "H",
    "h",
    "hB"
  ],
  NR: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  NU: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  NZ: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  OM: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  PA: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  PE: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  PF: [
    "H",
    "h",
    "hB"
  ],
  PG: [
    "h",
    "H"
  ],
  PH: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  PK: [
    "h",
    "hB",
    "H"
  ],
  PL: [
    "H",
    "h"
  ],
  PM: [
    "H",
    "hB"
  ],
  PN: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  PR: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  PS: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  PT: [
    "H",
    "hB"
  ],
  PW: [
    "h",
    "H"
  ],
  PY: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  QA: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  RE: [
    "H",
    "hB"
  ],
  RO: [
    "H",
    "hB"
  ],
  RS: [
    "H",
    "hB",
    "h"
  ],
  RU: [
    "H"
  ],
  RW: [
    "H",
    "h"
  ],
  SA: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  SB: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  SC: [
    "H",
    "h",
    "hB"
  ],
  SD: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  SE: [
    "H"
  ],
  SG: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  SH: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  SI: [
    "H",
    "hB"
  ],
  SJ: [
    "H"
  ],
  SK: [
    "H"
  ],
  SL: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  SM: [
    "H",
    "h",
    "hB"
  ],
  SN: [
    "H",
    "h",
    "hB"
  ],
  SO: [
    "h",
    "H"
  ],
  SR: [
    "H",
    "hB"
  ],
  SS: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  ST: [
    "H",
    "hB"
  ],
  SV: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  SX: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  SY: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  SZ: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  TA: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  TC: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  TD: [
    "h",
    "H",
    "hB"
  ],
  TF: [
    "H",
    "h",
    "hB"
  ],
  TG: [
    "H",
    "hB"
  ],
  TH: [
    "H",
    "h"
  ],
  TJ: [
    "H",
    "h"
  ],
  TL: [
    "H",
    "hB",
    "hb",
    "h"
  ],
  TM: [
    "H",
    "h"
  ],
  TN: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  TO: [
    "h",
    "H"
  ],
  TR: [
    "H",
    "hB"
  ],
  TT: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  TW: [
    "hB",
    "hb",
    "h",
    "H"
  ],
  TZ: [
    "hB",
    "hb",
    "H",
    "h"
  ],
  UA: [
    "H",
    "hB",
    "h"
  ],
  UG: [
    "hB",
    "hb",
    "H",
    "h"
  ],
  UM: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  US: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  UY: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  UZ: [
    "H",
    "hB",
    "h"
  ],
  VA: [
    "H",
    "h",
    "hB"
  ],
  VC: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  VE: [
    "h",
    "H",
    "hB",
    "hb"
  ],
  VG: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  VI: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  VN: [
    "H",
    "h"
  ],
  VU: [
    "h",
    "H"
  ],
  WF: [
    "H",
    "hB"
  ],
  WS: [
    "h",
    "H"
  ],
  XK: [
    "H",
    "hB",
    "h"
  ],
  YE: [
    "h",
    "hB",
    "hb",
    "H"
  ],
  YT: [
    "H",
    "hB"
  ],
  ZA: [
    "H",
    "h",
    "hb",
    "hB"
  ],
  ZM: [
    "h",
    "hb",
    "H",
    "hB"
  ],
  ZW: [
    "H",
    "h"
  ],
  "af-ZA": [
    "H",
    "h",
    "hB",
    "hb"
  ],
  "ar-001": [
    "h",
    "hB",
    "hb",
    "H"
  ],
  "ca-ES": [
    "H",
    "h",
    "hB"
  ],
  "en-001": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "en-HK": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "en-IL": [
    "H",
    "h",
    "hb",
    "hB"
  ],
  "en-MY": [
    "h",
    "hb",
    "H",
    "hB"
  ],
  "es-BR": [
    "H",
    "h",
    "hB",
    "hb"
  ],
  "es-ES": [
    "H",
    "h",
    "hB",
    "hb"
  ],
  "es-GQ": [
    "H",
    "h",
    "hB",
    "hb"
  ],
  "fr-CA": [
    "H",
    "h",
    "hB"
  ],
  "gl-ES": [
    "H",
    "h",
    "hB"
  ],
  "gu-IN": [
    "hB",
    "hb",
    "h",
    "H"
  ],
  "hi-IN": [
    "hB",
    "h",
    "H"
  ],
  "it-CH": [
    "H",
    "h",
    "hB"
  ],
  "it-IT": [
    "H",
    "h",
    "hB"
  ],
  "kn-IN": [
    "hB",
    "h",
    "H"
  ],
  "ml-IN": [
    "hB",
    "h",
    "H"
  ],
  "mr-IN": [
    "hB",
    "hb",
    "h",
    "H"
  ],
  "pa-IN": [
    "hB",
    "hb",
    "h",
    "H"
  ],
  "ta-IN": [
    "hB",
    "h",
    "hb",
    "H"
  ],
  "te-IN": [
    "hB",
    "h",
    "H"
  ],
  "zu-ZA": [
    "H",
    "hB",
    "hb",
    "h"
  ]
};
function Ql(e, t) {
  for (var r = "", n = 0; n < e.length; n++) {
    var s = e.charAt(n);
    if (s === "j") {
      for (var o = 0; n + 1 < e.length && e.charAt(n + 1) === s; )
        o++, n++;
      var f = 1 + (o & 1), c = o < 2 ? 1 : 3 + (o >> 1), d = "a", h = $l(t);
      for ((h == "H" || h == "k") && (c = 0); c-- > 0; )
        r += d;
      for (; f-- > 0; )
        r = h + r;
    } else s === "J" ? r += "H" : r += s;
  }
  return r;
}
function $l(e) {
  var t = e.hourCycle;
  if (t === void 0 && // @ts-ignore hourCycle(s) is not identified yet
  e.hourCycles && // @ts-ignore
  e.hourCycles.length && (t = e.hourCycles[0]), t)
    switch (t) {
      case "h24":
        return "k";
      case "h23":
        return "H";
      case "h12":
        return "h";
      case "h11":
        return "K";
      default:
        throw new Error("Invalid hourCycle");
    }
  var r = e.language, n;
  r !== "root" && (n = e.maximize().region);
  var s = vn[n || ""] || vn[r || ""] || vn["".concat(r, "-001")] || vn["001"];
  return s[0];
}
var Wn, eu = new RegExp("^".concat($a.source, "*")), tu = new RegExp("".concat($a.source, "*$"));
function ue(e, t) {
  return { start: e, end: t };
}
var ru = !!String.prototype.startsWith && "_a".startsWith("a", 1), nu = !!String.fromCodePoint, iu = !!Object.fromEntries, au = !!String.prototype.codePointAt, su = !!String.prototype.trimStart, ou = !!String.prototype.trimEnd, lu = !!Number.isSafeInteger, uu = lu ? Number.isSafeInteger : function(e) {
  return typeof e == "number" && isFinite(e) && Math.floor(e) === e && Math.abs(e) <= 9007199254740991;
}, si = !0;
try {
  var fu = is("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  si = ((Wn = fu.exec("a")) === null || Wn === void 0 ? void 0 : Wn[0]) === "a";
} catch {
  si = !1;
}
var ua = ru ? (
  // Native
  function(t, r, n) {
    return t.startsWith(r, n);
  }
) : (
  // For IE11
  function(t, r, n) {
    return t.slice(n, n + r.length) === r;
  }
), oi = nu ? String.fromCodePoint : (
  // IE11
  function() {
    for (var t = [], r = 0; r < arguments.length; r++)
      t[r] = arguments[r];
    for (var n = "", s = t.length, o = 0, f; s > o; ) {
      if (f = t[o++], f > 1114111)
        throw RangeError(f + " is not a valid code point");
      n += f < 65536 ? String.fromCharCode(f) : String.fromCharCode(((f -= 65536) >> 10) + 55296, f % 1024 + 56320);
    }
    return n;
  }
), fa = (
  // native
  iu ? Object.fromEntries : (
    // Ponyfill
    function(t) {
      for (var r = {}, n = 0, s = t; n < s.length; n++) {
        var o = s[n], f = o[0], c = o[1];
        r[f] = c;
      }
      return r;
    }
  )
), ns = au ? (
  // Native
  function(t, r) {
    return t.codePointAt(r);
  }
) : (
  // IE 11
  function(t, r) {
    var n = t.length;
    if (!(r < 0 || r >= n)) {
      var s = t.charCodeAt(r), o;
      return s < 55296 || s > 56319 || r + 1 === n || (o = t.charCodeAt(r + 1)) < 56320 || o > 57343 ? s : (s - 55296 << 10) + (o - 56320) + 65536;
    }
  }
), cu = su ? (
  // Native
  function(t) {
    return t.trimStart();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(eu, "");
  }
), hu = ou ? (
  // Native
  function(t) {
    return t.trimEnd();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(tu, "");
  }
);
function is(e, t) {
  return new RegExp(e, t);
}
var li;
if (si) {
  var ca = is("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  li = function(t, r) {
    var n;
    ca.lastIndex = r;
    var s = ca.exec(t);
    return (n = s[1]) !== null && n !== void 0 ? n : "";
  };
} else
  li = function(t, r) {
    for (var n = []; ; ) {
      var s = ns(t, r);
      if (s === void 0 || as(s) || gu(s))
        break;
      n.push(s), r += s >= 65536 ? 2 : 1;
    }
    return oi.apply(void 0, n);
  };
var du = (
  /** @class */
  (function() {
    function e(t, r) {
      r === void 0 && (r = {}), this.message = t, this.position = { offset: 0, line: 1, column: 1 }, this.ignoreTag = !!r.ignoreTag, this.locale = r.locale, this.requiresOtherClause = !!r.requiresOtherClause, this.shouldParseSkeletons = !!r.shouldParseSkeletons;
    }
    return e.prototype.parse = function() {
      if (this.offset() !== 0)
        throw Error("parser can only be used once");
      return this.parseMessage(0, "", !1);
    }, e.prototype.parseMessage = function(t, r, n) {
      for (var s = []; !this.isEOF(); ) {
        var o = this.char();
        if (o === 123) {
          var f = this.parseArgument(t, n);
          if (f.err)
            return f;
          s.push(f.val);
        } else {
          if (o === 125 && t > 0)
            break;
          if (o === 35 && (r === "plural" || r === "selectordinal")) {
            var c = this.clonePosition();
            this.bump(), s.push({
              type: Re.pound,
              location: ue(c, this.clonePosition())
            });
          } else if (o === 60 && !this.ignoreTag && this.peek() === 47) {
            if (n)
              break;
            return this.error(le.UNMATCHED_CLOSING_TAG, ue(this.clonePosition(), this.clonePosition()));
          } else if (o === 60 && !this.ignoreTag && ui(this.peek() || 0)) {
            var f = this.parseTag(t, r);
            if (f.err)
              return f;
            s.push(f.val);
          } else {
            var f = this.parseLiteral(t, r);
            if (f.err)
              return f;
            s.push(f.val);
          }
        }
      }
      return { val: s, err: null };
    }, e.prototype.parseTag = function(t, r) {
      var n = this.clonePosition();
      this.bump();
      var s = this.parseTagName();
      if (this.bumpSpace(), this.bumpIf("/>"))
        return {
          val: {
            type: Re.literal,
            value: "<".concat(s, "/>"),
            location: ue(n, this.clonePosition())
          },
          err: null
        };
      if (this.bumpIf(">")) {
        var o = this.parseMessage(t + 1, r, !0);
        if (o.err)
          return o;
        var f = o.val, c = this.clonePosition();
        if (this.bumpIf("</")) {
          if (this.isEOF() || !ui(this.char()))
            return this.error(le.INVALID_TAG, ue(c, this.clonePosition()));
          var d = this.clonePosition(), h = this.parseTagName();
          return s !== h ? this.error(le.UNMATCHED_CLOSING_TAG, ue(d, this.clonePosition())) : (this.bumpSpace(), this.bumpIf(">") ? {
            val: {
              type: Re.tag,
              value: s,
              children: f,
              location: ue(n, this.clonePosition())
            },
            err: null
          } : this.error(le.INVALID_TAG, ue(c, this.clonePosition())));
        } else
          return this.error(le.UNCLOSED_TAG, ue(n, this.clonePosition()));
      } else
        return this.error(le.INVALID_TAG, ue(n, this.clonePosition()));
    }, e.prototype.parseTagName = function() {
      var t = this.offset();
      for (this.bump(); !this.isEOF() && vu(this.char()); )
        this.bump();
      return this.message.slice(t, this.offset());
    }, e.prototype.parseLiteral = function(t, r) {
      for (var n = this.clonePosition(), s = ""; ; ) {
        var o = this.tryParseQuote(r);
        if (o) {
          s += o;
          continue;
        }
        var f = this.tryParseUnquoted(t, r);
        if (f) {
          s += f;
          continue;
        }
        var c = this.tryParseLeftAngleBracket();
        if (c) {
          s += c;
          continue;
        }
        break;
      }
      var d = ue(n, this.clonePosition());
      return {
        val: { type: Re.literal, value: s, location: d },
        err: null
      };
    }, e.prototype.tryParseLeftAngleBracket = function() {
      return !this.isEOF() && this.char() === 60 && (this.ignoreTag || // If at the opening tag or closing tag position, bail.
      !pu(this.peek() || 0)) ? (this.bump(), "<") : null;
    }, e.prototype.tryParseQuote = function(t) {
      if (this.isEOF() || this.char() !== 39)
        return null;
      switch (this.peek()) {
        case 39:
          return this.bump(), this.bump(), "'";
        // '{', '<', '>', '}'
        case 123:
        case 60:
        case 62:
        case 125:
          break;
        case 35:
          if (t === "plural" || t === "selectordinal")
            break;
          return null;
        default:
          return null;
      }
      this.bump();
      var r = [this.char()];
      for (this.bump(); !this.isEOF(); ) {
        var n = this.char();
        if (n === 39)
          if (this.peek() === 39)
            r.push(39), this.bump();
          else {
            this.bump();
            break;
          }
        else
          r.push(n);
        this.bump();
      }
      return oi.apply(void 0, r);
    }, e.prototype.tryParseUnquoted = function(t, r) {
      if (this.isEOF())
        return null;
      var n = this.char();
      return n === 60 || n === 123 || n === 35 && (r === "plural" || r === "selectordinal") || n === 125 && t > 0 ? null : (this.bump(), oi(n));
    }, e.prototype.parseArgument = function(t, r) {
      var n = this.clonePosition();
      if (this.bump(), this.bumpSpace(), this.isEOF())
        return this.error(le.EXPECT_ARGUMENT_CLOSING_BRACE, ue(n, this.clonePosition()));
      if (this.char() === 125)
        return this.bump(), this.error(le.EMPTY_ARGUMENT, ue(n, this.clonePosition()));
      var s = this.parseIdentifierIfPossible().value;
      if (!s)
        return this.error(le.MALFORMED_ARGUMENT, ue(n, this.clonePosition()));
      if (this.bumpSpace(), this.isEOF())
        return this.error(le.EXPECT_ARGUMENT_CLOSING_BRACE, ue(n, this.clonePosition()));
      switch (this.char()) {
        // Simple argument: `{name}`
        case 125:
          return this.bump(), {
            val: {
              type: Re.argument,
              // value does not include the opening and closing braces.
              value: s,
              location: ue(n, this.clonePosition())
            },
            err: null
          };
        // Argument with options: `{name, format, ...}`
        case 44:
          return this.bump(), this.bumpSpace(), this.isEOF() ? this.error(le.EXPECT_ARGUMENT_CLOSING_BRACE, ue(n, this.clonePosition())) : this.parseArgumentOptions(t, r, s, n);
        default:
          return this.error(le.MALFORMED_ARGUMENT, ue(n, this.clonePosition()));
      }
    }, e.prototype.parseIdentifierIfPossible = function() {
      var t = this.clonePosition(), r = this.offset(), n = li(this.message, r), s = r + n.length;
      this.bumpTo(s);
      var o = this.clonePosition(), f = ue(t, o);
      return { value: n, location: f };
    }, e.prototype.parseArgumentOptions = function(t, r, n, s) {
      var o, f = this.clonePosition(), c = this.parseIdentifierIfPossible().value, d = this.clonePosition();
      switch (c) {
        case "":
          return this.error(le.EXPECT_ARGUMENT_TYPE, ue(f, d));
        case "number":
        case "date":
        case "time": {
          this.bumpSpace();
          var h = null;
          if (this.bumpIf(",")) {
            this.bumpSpace();
            var m = this.clonePosition(), T = this.parseSimpleArgStyleIfPossible();
            if (T.err)
              return T;
            var w = hu(T.val);
            if (w.length === 0)
              return this.error(le.EXPECT_ARGUMENT_STYLE, ue(this.clonePosition(), this.clonePosition()));
            var L = ue(m, this.clonePosition());
            h = { style: w, styleLocation: L };
          }
          var R = this.tryParseArgumentClose(s);
          if (R.err)
            return R;
          var P = ue(s, this.clonePosition());
          if (h && ua(h?.style, "::", 0)) {
            var H = cu(h.style.slice(2));
            if (c === "number") {
              var T = this.parseNumberSkeletonFromString(H, h.styleLocation);
              return T.err ? T : {
                val: { type: Re.number, value: n, location: P, style: T.val },
                err: null
              };
            } else {
              if (H.length === 0)
                return this.error(le.EXPECT_DATE_TIME_SKELETON, P);
              var _ = H;
              this.locale && (_ = Ql(H, this.locale));
              var w = {
                type: Fr.dateTime,
                pattern: _,
                location: h.styleLocation,
                parsedOptions: this.shouldParseSkeletons ? Xl(_) : {}
              }, y = c === "date" ? Re.date : Re.time;
              return {
                val: { type: y, value: n, location: P, style: w },
                err: null
              };
            }
          }
          return {
            val: {
              type: c === "number" ? Re.number : c === "date" ? Re.date : Re.time,
              value: n,
              location: P,
              style: (o = h?.style) !== null && o !== void 0 ? o : null
            },
            err: null
          };
        }
        case "plural":
        case "selectordinal":
        case "select": {
          var S = this.clonePosition();
          if (this.bumpSpace(), !this.bumpIf(","))
            return this.error(le.EXPECT_SELECT_ARGUMENT_OPTIONS, ue(S, he({}, S)));
          this.bumpSpace();
          var E = this.parseIdentifierIfPossible(), b = 0;
          if (c !== "select" && E.value === "offset") {
            if (!this.bumpIf(":"))
              return this.error(le.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, ue(this.clonePosition(), this.clonePosition()));
            this.bumpSpace();
            var T = this.tryParseDecimalInteger(le.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, le.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE);
            if (T.err)
              return T;
            this.bumpSpace(), E = this.parseIdentifierIfPossible(), b = T.val;
          }
          var A = this.tryParsePluralOrSelectOptions(t, c, r, E);
          if (A.err)
            return A;
          var R = this.tryParseArgumentClose(s);
          if (R.err)
            return R;
          var N = ue(s, this.clonePosition());
          return c === "select" ? {
            val: {
              type: Re.select,
              value: n,
              options: fa(A.val),
              location: N
            },
            err: null
          } : {
            val: {
              type: Re.plural,
              value: n,
              options: fa(A.val),
              offset: b,
              pluralType: c === "plural" ? "cardinal" : "ordinal",
              location: N
            },
            err: null
          };
        }
        default:
          return this.error(le.INVALID_ARGUMENT_TYPE, ue(f, d));
      }
    }, e.prototype.tryParseArgumentClose = function(t) {
      return this.isEOF() || this.char() !== 125 ? this.error(le.EXPECT_ARGUMENT_CLOSING_BRACE, ue(t, this.clonePosition())) : (this.bump(), { val: !0, err: null });
    }, e.prototype.parseSimpleArgStyleIfPossible = function() {
      for (var t = 0, r = this.clonePosition(); !this.isEOF(); ) {
        var n = this.char();
        switch (n) {
          case 39: {
            this.bump();
            var s = this.clonePosition();
            if (!this.bumpUntil("'"))
              return this.error(le.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE, ue(s, this.clonePosition()));
            this.bump();
            break;
          }
          case 123: {
            t += 1, this.bump();
            break;
          }
          case 125: {
            if (t > 0)
              t -= 1;
            else
              return {
                val: this.message.slice(r.offset, this.offset()),
                err: null
              };
            break;
          }
          default:
            this.bump();
            break;
        }
      }
      return {
        val: this.message.slice(r.offset, this.offset()),
        err: null
      };
    }, e.prototype.parseNumberSkeletonFromString = function(t, r) {
      var n = [];
      try {
        n = ql(t);
      } catch {
        return this.error(le.INVALID_NUMBER_SKELETON, r);
      }
      return {
        val: {
          type: Fr.number,
          tokens: n,
          location: r,
          parsedOptions: this.shouldParseSkeletons ? Kl(n) : {}
        },
        err: null
      };
    }, e.prototype.tryParsePluralOrSelectOptions = function(t, r, n, s) {
      for (var o, f = !1, c = [], d = /* @__PURE__ */ new Set(), h = s.value, m = s.location; ; ) {
        if (h.length === 0) {
          var T = this.clonePosition();
          if (r !== "select" && this.bumpIf("=")) {
            var w = this.tryParseDecimalInteger(le.EXPECT_PLURAL_ARGUMENT_SELECTOR, le.INVALID_PLURAL_ARGUMENT_SELECTOR);
            if (w.err)
              return w;
            m = ue(T, this.clonePosition()), h = this.message.slice(T.offset, this.offset());
          } else
            break;
        }
        if (d.has(h))
          return this.error(r === "select" ? le.DUPLICATE_SELECT_ARGUMENT_SELECTOR : le.DUPLICATE_PLURAL_ARGUMENT_SELECTOR, m);
        h === "other" && (f = !0), this.bumpSpace();
        var L = this.clonePosition();
        if (!this.bumpIf("{"))
          return this.error(r === "select" ? le.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT : le.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT, ue(this.clonePosition(), this.clonePosition()));
        var R = this.parseMessage(t + 1, r, n);
        if (R.err)
          return R;
        var P = this.tryParseArgumentClose(L);
        if (P.err)
          return P;
        c.push([
          h,
          {
            value: R.val,
            location: ue(L, this.clonePosition())
          }
        ]), d.add(h), this.bumpSpace(), o = this.parseIdentifierIfPossible(), h = o.value, m = o.location;
      }
      return c.length === 0 ? this.error(r === "select" ? le.EXPECT_SELECT_ARGUMENT_SELECTOR : le.EXPECT_PLURAL_ARGUMENT_SELECTOR, ue(this.clonePosition(), this.clonePosition())) : this.requiresOtherClause && !f ? this.error(le.MISSING_OTHER_CLAUSE, ue(this.clonePosition(), this.clonePosition())) : { val: c, err: null };
    }, e.prototype.tryParseDecimalInteger = function(t, r) {
      var n = 1, s = this.clonePosition();
      this.bumpIf("+") || this.bumpIf("-") && (n = -1);
      for (var o = !1, f = 0; !this.isEOF(); ) {
        var c = this.char();
        if (c >= 48 && c <= 57)
          o = !0, f = f * 10 + (c - 48), this.bump();
        else
          break;
      }
      var d = ue(s, this.clonePosition());
      return o ? (f *= n, uu(f) ? { val: f, err: null } : this.error(r, d)) : this.error(t, d);
    }, e.prototype.offset = function() {
      return this.position.offset;
    }, e.prototype.isEOF = function() {
      return this.offset() === this.message.length;
    }, e.prototype.clonePosition = function() {
      return {
        offset: this.position.offset,
        line: this.position.line,
        column: this.position.column
      };
    }, e.prototype.char = function() {
      var t = this.position.offset;
      if (t >= this.message.length)
        throw Error("out of bound");
      var r = ns(this.message, t);
      if (r === void 0)
        throw Error("Offset ".concat(t, " is at invalid UTF-16 code unit boundary"));
      return r;
    }, e.prototype.error = function(t, r) {
      return {
        val: null,
        err: {
          kind: t,
          message: this.message,
          location: r
        }
      };
    }, e.prototype.bump = function() {
      if (!this.isEOF()) {
        var t = this.char();
        t === 10 ? (this.position.line += 1, this.position.column = 1, this.position.offset += 1) : (this.position.column += 1, this.position.offset += t < 65536 ? 1 : 2);
      }
    }, e.prototype.bumpIf = function(t) {
      if (ua(this.message, t, this.offset())) {
        for (var r = 0; r < t.length; r++)
          this.bump();
        return !0;
      }
      return !1;
    }, e.prototype.bumpUntil = function(t) {
      var r = this.offset(), n = this.message.indexOf(t, r);
      return n >= 0 ? (this.bumpTo(n), !0) : (this.bumpTo(this.message.length), !1);
    }, e.prototype.bumpTo = function(t) {
      if (this.offset() > t)
        throw Error("targetOffset ".concat(t, " must be greater than or equal to the current offset ").concat(this.offset()));
      for (t = Math.min(t, this.message.length); ; ) {
        var r = this.offset();
        if (r === t)
          break;
        if (r > t)
          throw Error("targetOffset ".concat(t, " is at invalid UTF-16 code unit boundary"));
        if (this.bump(), this.isEOF())
          break;
      }
    }, e.prototype.bumpSpace = function() {
      for (; !this.isEOF() && as(this.char()); )
        this.bump();
    }, e.prototype.peek = function() {
      if (this.isEOF())
        return null;
      var t = this.char(), r = this.offset(), n = this.message.charCodeAt(r + (t >= 65536 ? 2 : 1));
      return n ?? null;
    }, e;
  })()
);
function ui(e) {
  return e >= 97 && e <= 122 || e >= 65 && e <= 90;
}
function pu(e) {
  return ui(e) || e === 47;
}
function vu(e) {
  return e === 45 || e === 46 || e >= 48 && e <= 57 || e === 95 || e >= 97 && e <= 122 || e >= 65 && e <= 90 || e == 183 || e >= 192 && e <= 214 || e >= 216 && e <= 246 || e >= 248 && e <= 893 || e >= 895 && e <= 8191 || e >= 8204 && e <= 8205 || e >= 8255 && e <= 8256 || e >= 8304 && e <= 8591 || e >= 11264 && e <= 12271 || e >= 12289 && e <= 55295 || e >= 63744 && e <= 64975 || e >= 65008 && e <= 65533 || e >= 65536 && e <= 983039;
}
function as(e) {
  return e >= 9 && e <= 13 || e === 32 || e === 133 || e >= 8206 && e <= 8207 || e === 8232 || e === 8233;
}
function gu(e) {
  return e >= 33 && e <= 35 || e === 36 || e >= 37 && e <= 39 || e === 40 || e === 41 || e === 42 || e === 43 || e === 44 || e === 45 || e >= 46 && e <= 47 || e >= 58 && e <= 59 || e >= 60 && e <= 62 || e >= 63 && e <= 64 || e === 91 || e === 92 || e === 93 || e === 94 || e === 96 || e === 123 || e === 124 || e === 125 || e === 126 || e === 161 || e >= 162 && e <= 165 || e === 166 || e === 167 || e === 169 || e === 171 || e === 172 || e === 174 || e === 176 || e === 177 || e === 182 || e === 187 || e === 191 || e === 215 || e === 247 || e >= 8208 && e <= 8213 || e >= 8214 && e <= 8215 || e === 8216 || e === 8217 || e === 8218 || e >= 8219 && e <= 8220 || e === 8221 || e === 8222 || e === 8223 || e >= 8224 && e <= 8231 || e >= 8240 && e <= 8248 || e === 8249 || e === 8250 || e >= 8251 && e <= 8254 || e >= 8257 && e <= 8259 || e === 8260 || e === 8261 || e === 8262 || e >= 8263 && e <= 8273 || e === 8274 || e === 8275 || e >= 8277 && e <= 8286 || e >= 8592 && e <= 8596 || e >= 8597 && e <= 8601 || e >= 8602 && e <= 8603 || e >= 8604 && e <= 8607 || e === 8608 || e >= 8609 && e <= 8610 || e === 8611 || e >= 8612 && e <= 8613 || e === 8614 || e >= 8615 && e <= 8621 || e === 8622 || e >= 8623 && e <= 8653 || e >= 8654 && e <= 8655 || e >= 8656 && e <= 8657 || e === 8658 || e === 8659 || e === 8660 || e >= 8661 && e <= 8691 || e >= 8692 && e <= 8959 || e >= 8960 && e <= 8967 || e === 8968 || e === 8969 || e === 8970 || e === 8971 || e >= 8972 && e <= 8991 || e >= 8992 && e <= 8993 || e >= 8994 && e <= 9e3 || e === 9001 || e === 9002 || e >= 9003 && e <= 9083 || e === 9084 || e >= 9085 && e <= 9114 || e >= 9115 && e <= 9139 || e >= 9140 && e <= 9179 || e >= 9180 && e <= 9185 || e >= 9186 && e <= 9254 || e >= 9255 && e <= 9279 || e >= 9280 && e <= 9290 || e >= 9291 && e <= 9311 || e >= 9472 && e <= 9654 || e === 9655 || e >= 9656 && e <= 9664 || e === 9665 || e >= 9666 && e <= 9719 || e >= 9720 && e <= 9727 || e >= 9728 && e <= 9838 || e === 9839 || e >= 9840 && e <= 10087 || e === 10088 || e === 10089 || e === 10090 || e === 10091 || e === 10092 || e === 10093 || e === 10094 || e === 10095 || e === 10096 || e === 10097 || e === 10098 || e === 10099 || e === 10100 || e === 10101 || e >= 10132 && e <= 10175 || e >= 10176 && e <= 10180 || e === 10181 || e === 10182 || e >= 10183 && e <= 10213 || e === 10214 || e === 10215 || e === 10216 || e === 10217 || e === 10218 || e === 10219 || e === 10220 || e === 10221 || e === 10222 || e === 10223 || e >= 10224 && e <= 10239 || e >= 10240 && e <= 10495 || e >= 10496 && e <= 10626 || e === 10627 || e === 10628 || e === 10629 || e === 10630 || e === 10631 || e === 10632 || e === 10633 || e === 10634 || e === 10635 || e === 10636 || e === 10637 || e === 10638 || e === 10639 || e === 10640 || e === 10641 || e === 10642 || e === 10643 || e === 10644 || e === 10645 || e === 10646 || e === 10647 || e === 10648 || e >= 10649 && e <= 10711 || e === 10712 || e === 10713 || e === 10714 || e === 10715 || e >= 10716 && e <= 10747 || e === 10748 || e === 10749 || e >= 10750 && e <= 11007 || e >= 11008 && e <= 11055 || e >= 11056 && e <= 11076 || e >= 11077 && e <= 11078 || e >= 11079 && e <= 11084 || e >= 11085 && e <= 11123 || e >= 11124 && e <= 11125 || e >= 11126 && e <= 11157 || e === 11158 || e >= 11159 && e <= 11263 || e >= 11776 && e <= 11777 || e === 11778 || e === 11779 || e === 11780 || e === 11781 || e >= 11782 && e <= 11784 || e === 11785 || e === 11786 || e === 11787 || e === 11788 || e === 11789 || e >= 11790 && e <= 11798 || e === 11799 || e >= 11800 && e <= 11801 || e === 11802 || e === 11803 || e === 11804 || e === 11805 || e >= 11806 && e <= 11807 || e === 11808 || e === 11809 || e === 11810 || e === 11811 || e === 11812 || e === 11813 || e === 11814 || e === 11815 || e === 11816 || e === 11817 || e >= 11818 && e <= 11822 || e === 11823 || e >= 11824 && e <= 11833 || e >= 11834 && e <= 11835 || e >= 11836 && e <= 11839 || e === 11840 || e === 11841 || e === 11842 || e >= 11843 && e <= 11855 || e >= 11856 && e <= 11857 || e === 11858 || e >= 11859 && e <= 11903 || e >= 12289 && e <= 12291 || e === 12296 || e === 12297 || e === 12298 || e === 12299 || e === 12300 || e === 12301 || e === 12302 || e === 12303 || e === 12304 || e === 12305 || e >= 12306 && e <= 12307 || e === 12308 || e === 12309 || e === 12310 || e === 12311 || e === 12312 || e === 12313 || e === 12314 || e === 12315 || e === 12316 || e === 12317 || e >= 12318 && e <= 12319 || e === 12320 || e === 12336 || e === 64830 || e === 64831 || e >= 65093 && e <= 65094;
}
function fi(e) {
  e.forEach(function(t) {
    if (delete t.location, Za(t) || Ja(t))
      for (var r in t.options)
        delete t.options[r].location, fi(t.options[r].value);
    else Wa(t) && Qa(t.style) || (qa(t) || Ya(t)) && ai(t.style) ? delete t.style.location : Ka(t) && fi(t.children);
  });
}
function mu(e, t) {
  t === void 0 && (t = {}), t = he({ shouldParseSkeletons: !0, requiresOtherClause: !0 }, t);
  var r = new du(e, t).parse();
  if (r.err) {
    var n = SyntaxError(le[r.err.kind]);
    throw n.location = r.err.location, n.originalMessage = r.err.message, n;
  }
  return t?.captureLocation || fi(r.val), r.val;
}
var jr;
(function(e) {
  e.MISSING_VALUE = "MISSING_VALUE", e.INVALID_VALUE = "INVALID_VALUE", e.MISSING_INTL_API = "MISSING_INTL_API";
})(jr || (jr = {}));
var In = (
  /** @class */
  (function(e) {
    Hn(t, e);
    function t(r, n, s) {
      var o = e.call(this, r) || this;
      return o.code = n, o.originalMessage = s, o;
    }
    return t.prototype.toString = function() {
      return "[formatjs Error: ".concat(this.code, "] ").concat(this.message);
    }, t;
  })(Error)
), ha = (
  /** @class */
  (function(e) {
    Hn(t, e);
    function t(r, n, s, o) {
      return e.call(this, 'Invalid values for "'.concat(r, '": "').concat(n, '". Options are "').concat(Object.keys(s).join('", "'), '"'), jr.INVALID_VALUE, o) || this;
    }
    return t;
  })(In)
), _u = (
  /** @class */
  (function(e) {
    Hn(t, e);
    function t(r, n, s) {
      return e.call(this, 'Value for "'.concat(r, '" must be of type ').concat(n), jr.INVALID_VALUE, s) || this;
    }
    return t;
  })(In)
), bu = (
  /** @class */
  (function(e) {
    Hn(t, e);
    function t(r, n) {
      return e.call(this, 'The intl string context variable "'.concat(r, '" was not provided to the string "').concat(n, '"'), jr.MISSING_VALUE, n) || this;
    }
    return t;
  })(In)
), vt;
(function(e) {
  e[e.literal = 0] = "literal", e[e.object = 1] = "object";
})(vt || (vt = {}));
function yu(e) {
  return e.length < 2 ? e : e.reduce(function(t, r) {
    var n = t[t.length - 1];
    return !n || n.type !== vt.literal || r.type !== vt.literal ? t.push(r) : n.value += r.value, t;
  }, []);
}
function xu(e) {
  return typeof e == "function";
}
function _n(e, t, r, n, s, o, f) {
  if (e.length === 1 && aa(e[0]))
    return [
      {
        type: vt.literal,
        value: e[0].value
      }
    ];
  for (var c = [], d = 0, h = e; d < h.length; d++) {
    var m = h[d];
    if (aa(m)) {
      c.push({
        type: vt.literal,
        value: m.value
      });
      continue;
    }
    if (Vl(m)) {
      typeof o == "number" && c.push({
        type: vt.literal,
        value: r.getNumberFormat(t).format(o)
      });
      continue;
    }
    var T = m.value;
    if (!(s && T in s))
      throw new bu(T, f);
    var w = s[T];
    if (Ul(m)) {
      (!w || typeof w == "string" || typeof w == "number") && (w = typeof w == "string" || typeof w == "number" ? String(w) : ""), c.push({
        type: typeof w == "string" ? vt.literal : vt.object,
        value: w
      });
      continue;
    }
    if (qa(m)) {
      var L = typeof m.style == "string" ? n.date[m.style] : ai(m.style) ? m.style.parsedOptions : void 0;
      c.push({
        type: vt.literal,
        value: r.getDateTimeFormat(t, L).format(w)
      });
      continue;
    }
    if (Ya(m)) {
      var L = typeof m.style == "string" ? n.time[m.style] : ai(m.style) ? m.style.parsedOptions : n.time.medium;
      c.push({
        type: vt.literal,
        value: r.getDateTimeFormat(t, L).format(w)
      });
      continue;
    }
    if (Wa(m)) {
      var L = typeof m.style == "string" ? n.number[m.style] : Qa(m.style) ? m.style.parsedOptions : void 0;
      L && L.scale && (w = w * (L.scale || 1)), c.push({
        type: vt.literal,
        value: r.getNumberFormat(t, L).format(w)
      });
      continue;
    }
    if (Ka(m)) {
      var R = m.children, P = m.value, H = s[P];
      if (!xu(H))
        throw new _u(P, "function", f);
      var _ = _n(R, t, r, n, s, o), y = H(_.map(function(b) {
        return b.value;
      }));
      Array.isArray(y) || (y = [y]), c.push.apply(c, y.map(function(b) {
        return {
          type: typeof b == "string" ? vt.literal : vt.object,
          value: b
        };
      }));
    }
    if (Za(m)) {
      var S = m.options[w] || m.options.other;
      if (!S)
        throw new ha(m.value, w, Object.keys(m.options), f);
      c.push.apply(c, _n(S.value, t, r, n, s));
      continue;
    }
    if (Ja(m)) {
      var S = m.options["=".concat(w)];
      if (!S) {
        if (!Intl.PluralRules)
          throw new In(`Intl.PluralRules is not available in this environment.
Try polyfilling it using "@formatjs/intl-pluralrules"
`, jr.MISSING_INTL_API, f);
        var E = r.getPluralRules(t, { type: m.pluralType }).select(w - (m.offset || 0));
        S = m.options[E] || m.options.other;
      }
      if (!S)
        throw new ha(m.value, w, Object.keys(m.options), f);
      c.push.apply(c, _n(S.value, t, r, n, s, w - (m.offset || 0)));
      continue;
    }
  }
  return yu(c);
}
function wu(e, t) {
  return t ? he(he(he({}, e || {}), t || {}), Object.keys(e).reduce(function(r, n) {
    return r[n] = he(he({}, e[n]), t[n] || {}), r;
  }, {})) : e;
}
function Eu(e, t) {
  return t ? Object.keys(e).reduce(function(r, n) {
    return r[n] = wu(e[n], t[n]), r;
  }, he({}, e)) : e;
}
function qn(e) {
  return {
    create: function() {
      return {
        get: function(t) {
          return e[t];
        },
        set: function(t, r) {
          e[t] = r;
        }
      };
    }
  };
}
function Su(e) {
  return e === void 0 && (e = {
    number: {},
    dateTime: {},
    pluralRules: {}
  }), {
    getNumberFormat: zn(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.NumberFormat).bind.apply(t, Vn([void 0], r, !1)))();
    }, {
      cache: qn(e.number),
      strategy: Xn.variadic
    }),
    getDateTimeFormat: zn(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.DateTimeFormat).bind.apply(t, Vn([void 0], r, !1)))();
    }, {
      cache: qn(e.dateTime),
      strategy: Xn.variadic
    }),
    getPluralRules: zn(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.PluralRules).bind.apply(t, Vn([void 0], r, !1)))();
    }, {
      cache: qn(e.pluralRules),
      strategy: Xn.variadic
    })
  };
}
var Tu = (
  /** @class */
  (function() {
    function e(t, r, n, s) {
      r === void 0 && (r = e.defaultLocale);
      var o = this;
      if (this.formatterCache = {
        number: {},
        dateTime: {},
        pluralRules: {}
      }, this.format = function(d) {
        var h = o.formatToParts(d);
        if (h.length === 1)
          return h[0].value;
        var m = h.reduce(function(T, w) {
          return !T.length || w.type !== vt.literal || typeof T[T.length - 1] != "string" ? T.push(w.value) : T[T.length - 1] += w.value, T;
        }, []);
        return m.length <= 1 ? m[0] || "" : m;
      }, this.formatToParts = function(d) {
        return _n(o.ast, o.locales, o.formatters, o.formats, d, void 0, o.message);
      }, this.resolvedOptions = function() {
        var d;
        return {
          locale: ((d = o.resolvedLocale) === null || d === void 0 ? void 0 : d.toString()) || Intl.NumberFormat.supportedLocalesOf(o.locales)[0]
        };
      }, this.getAst = function() {
        return o.ast;
      }, this.locales = r, this.resolvedLocale = e.resolveLocale(r), typeof t == "string") {
        if (this.message = t, !e.__parse)
          throw new TypeError("IntlMessageFormat.__parse must be set to process `message` of type `string`");
        var f = s || {};
        f.formatters;
        var c = Ll(f, ["formatters"]);
        this.ast = e.__parse(t, he(he({}, c), { locale: this.resolvedLocale }));
      } else
        this.ast = t;
      if (!Array.isArray(this.ast))
        throw new TypeError("A message must be provided as a String or AST.");
      this.formats = Eu(e.formats, n), this.formatters = s && s.formatters || Su(this.formatterCache);
    }
    return Object.defineProperty(e, "defaultLocale", {
      get: function() {
        return e.memoizedDefaultLocale || (e.memoizedDefaultLocale = new Intl.NumberFormat().resolvedOptions().locale), e.memoizedDefaultLocale;
      },
      enumerable: !1,
      configurable: !0
    }), e.memoizedDefaultLocale = null, e.resolveLocale = function(t) {
      if (!(typeof Intl.Locale > "u")) {
        var r = Intl.NumberFormat.supportedLocalesOf(t);
        return r.length > 0 ? new Intl.Locale(r[0]) : new Intl.Locale(typeof t == "string" ? t : t[0]);
      }
    }, e.__parse = mu, e.formats = {
      number: {
        integer: {
          maximumFractionDigits: 0
        },
        currency: {
          style: "currency"
        },
        percent: {
          style: "percent"
        }
      },
      date: {
        short: {
          month: "numeric",
          day: "numeric",
          year: "2-digit"
        },
        medium: {
          month: "short",
          day: "numeric",
          year: "numeric"
        },
        long: {
          month: "long",
          day: "numeric",
          year: "numeric"
        },
        full: {
          weekday: "long",
          month: "long",
          day: "numeric",
          year: "numeric"
        }
      },
      time: {
        short: {
          hour: "numeric",
          minute: "numeric"
        },
        medium: {
          hour: "numeric",
          minute: "numeric",
          second: "numeric"
        },
        long: {
          hour: "numeric",
          minute: "numeric",
          second: "numeric",
          timeZoneName: "short"
        },
        full: {
          hour: "numeric",
          minute: "numeric",
          second: "numeric",
          timeZoneName: "short"
        }
      }
    }, e;
  })()
);
function Mu(e, t) {
  if (t == null)
    return;
  if (t in e)
    return e[t];
  const r = t.split(".");
  let n = e;
  for (let s = 0; s < r.length; s++)
    if (typeof n == "object") {
      if (s > 0) {
        const o = r.slice(s, r.length).join(".");
        if (o in n) {
          n = n[o];
          break;
        }
      }
      n = n[r[s]];
    } else
      n = void 0;
  return n;
}
const pr = {}, Au = (e, t, r) => r && (t in pr || (pr[t] = {}), e in pr[t] || (pr[t][e] = r), r), ss = (e, t) => {
  if (t == null)
    return;
  if (t in pr && e in pr[t])
    return pr[t][e];
  const r = Pn(t);
  for (let n = 0; n < r.length; n++) {
    const s = r[n], o = Nu(s, e);
    if (o)
      return Au(e, t, o);
  }
};
let Si;
const sn = an({});
function Ou(e) {
  return Si[e] || null;
}
function os(e) {
  return e in Si;
}
function Nu(e, t) {
  if (!os(e))
    return null;
  const r = Ou(e);
  return Mu(r, t);
}
function Hu(e) {
  if (e == null)
    return;
  const t = Pn(e);
  for (let r = 0; r < t.length; r++) {
    const n = t[r];
    if (os(n))
      return n;
  }
}
function Iu(e, ...t) {
  delete pr[e], sn.update((r) => (r[e] = Cl.all([r[e] || {}, ...t]), r));
}
Xr(
  [sn],
  ([e]) => Object.keys(e)
);
sn.subscribe((e) => Si = e);
const bn = {};
function Pu(e, t) {
  bn[e].delete(t), bn[e].size === 0 && delete bn[e];
}
function ls(e) {
  return bn[e];
}
function Cu(e) {
  return Pn(e).map((t) => {
    const r = ls(t);
    return [t, r ? [...r] : []];
  }).filter(([, t]) => t.length > 0);
}
function ci(e) {
  return e == null ? !1 : Pn(e).some(
    (t) => {
      var r;
      return (r = ls(t)) == null ? void 0 : r.size;
    }
  );
}
function Lu(e, t) {
  return Promise.all(
    t.map((n) => (Pu(e, n), n().then((s) => s.default || s)))
  ).then((n) => Iu(e, ...n));
}
const Qr = {};
function us(e) {
  if (!ci(e))
    return e in Qr ? Qr[e] : Promise.resolve();
  const t = Cu(e);
  return Qr[e] = Promise.all(
    t.map(
      ([r, n]) => Lu(r, n)
    )
  ).then(() => {
    if (ci(e))
      return us(e);
    delete Qr[e];
  }), Qr[e];
}
const Bu = {
  number: {
    scientific: { notation: "scientific" },
    engineering: { notation: "engineering" },
    compactLong: { notation: "compact", compactDisplay: "long" },
    compactShort: { notation: "compact", compactDisplay: "short" }
  },
  date: {
    short: { month: "numeric", day: "numeric", year: "2-digit" },
    medium: { month: "short", day: "numeric", year: "numeric" },
    long: { month: "long", day: "numeric", year: "numeric" },
    full: { weekday: "long", month: "long", day: "numeric", year: "numeric" }
  },
  time: {
    short: { hour: "numeric", minute: "numeric" },
    medium: { hour: "numeric", minute: "numeric", second: "numeric" },
    long: {
      hour: "numeric",
      minute: "numeric",
      second: "numeric",
      timeZoneName: "short"
    },
    full: {
      hour: "numeric",
      minute: "numeric",
      second: "numeric",
      timeZoneName: "short"
    }
  }
}, Ru = {
  fallbackLocale: null,
  loadingDelay: 200,
  formats: Bu,
  warnOnMissingMessages: !0,
  handleMissingMessage: void 0,
  ignoreTag: !0
}, ku = Ru;
function Ur() {
  return ku;
}
const Yn = an(!1);
var Gu = Object.defineProperty, Du = Object.defineProperties, Fu = Object.getOwnPropertyDescriptors, da = Object.getOwnPropertySymbols, ju = Object.prototype.hasOwnProperty, Uu = Object.prototype.propertyIsEnumerable, pa = (e, t, r) => t in e ? Gu(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, Vu = (e, t) => {
  for (var r in t || (t = {}))
    ju.call(t, r) && pa(e, r, t[r]);
  if (da)
    for (var r of da(t))
      Uu.call(t, r) && pa(e, r, t[r]);
  return e;
}, zu = (e, t) => Du(e, Fu(t));
let hi;
const wn = an(null);
function va(e) {
  return e.split("-").map((t, r, n) => n.slice(0, r + 1).join("-")).reverse();
}
function Pn(e, t = Ur().fallbackLocale) {
  const r = va(e);
  return t ? [.../* @__PURE__ */ new Set([...r, ...va(t)])] : r;
}
function Hr() {
  return hi ?? void 0;
}
wn.subscribe((e) => {
  hi = e ?? void 0, typeof window < "u" && e != null && document.documentElement.setAttribute("lang", e);
});
const Xu = (e) => {
  if (e && Hu(e) && ci(e)) {
    const { loadingDelay: t } = Ur();
    let r;
    return typeof window < "u" && Hr() != null && t ? r = window.setTimeout(
      () => Yn.set(!0),
      t
    ) : Yn.set(!0), us(e).then(() => {
      wn.set(e);
    }).finally(() => {
      clearTimeout(r), Yn.set(!1);
    });
  }
  return wn.set(e);
}, Wr = zu(Vu({}, wn), {
  set: Xu
}), Cn = (e) => {
  const t = /* @__PURE__ */ Object.create(null);
  return (n) => {
    const s = JSON.stringify(n);
    return s in t ? t[s] : t[s] = e(n);
  };
};
var Wu = Object.defineProperty, En = Object.getOwnPropertySymbols, fs = Object.prototype.hasOwnProperty, cs = Object.prototype.propertyIsEnumerable, ga = (e, t, r) => t in e ? Wu(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, Ti = (e, t) => {
  for (var r in t || (t = {}))
    fs.call(t, r) && ga(e, r, t[r]);
  if (En)
    for (var r of En(t))
      cs.call(t, r) && ga(e, r, t[r]);
  return e;
}, qr = (e, t) => {
  var r = {};
  for (var n in e)
    fs.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && En)
    for (var n of En(e))
      t.indexOf(n) < 0 && cs.call(e, n) && (r[n] = e[n]);
  return r;
};
const rn = (e, t) => {
  const { formats: r } = Ur();
  if (e in r && t in r[e])
    return r[e][t];
  throw new Error(`[svelte-i18n] Unknown "${t}" ${e} format.`);
}, qu = Cn(
  (e) => {
    var t = e, { locale: r, format: n } = t, s = qr(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format numbers');
    return n && (s = rn("number", n)), new Intl.NumberFormat(r, s);
  }
), Yu = Cn(
  (e) => {
    var t = e, { locale: r, format: n } = t, s = qr(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format dates');
    return n ? s = rn("date", n) : Object.keys(s).length === 0 && (s = rn("date", "short")), new Intl.DateTimeFormat(r, s);
  }
), Zu = Cn(
  (e) => {
    var t = e, { locale: r, format: n } = t, s = qr(t, ["locale", "format"]);
    if (r == null)
      throw new Error(
        '[svelte-i18n] A "locale" must be set to format time values'
      );
    return n ? s = rn("time", n) : Object.keys(s).length === 0 && (s = rn("time", "short")), new Intl.DateTimeFormat(r, s);
  }
), Ju = (e = {}) => {
  var t = e, {
    locale: r = Hr()
  } = t, n = qr(t, [
    "locale"
  ]);
  return qu(Ti({ locale: r }, n));
}, Ku = (e = {}) => {
  var t = e, {
    locale: r = Hr()
  } = t, n = qr(t, [
    "locale"
  ]);
  return Yu(Ti({ locale: r }, n));
}, Qu = (e = {}) => {
  var t = e, {
    locale: r = Hr()
  } = t, n = qr(t, [
    "locale"
  ]);
  return Zu(Ti({ locale: r }, n));
}, $u = Cn(
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  (e, t = Hr()) => new Tu(e, t, Ur().formats, {
    ignoreTag: Ur().ignoreTag
  })
), ef = (e, t = {}) => {
  var r, n, s, o;
  let f = t;
  typeof e == "object" && (f = e, e = f.id);
  const {
    values: c,
    locale: d = Hr(),
    default: h
  } = f;
  if (d == null)
    throw new Error(
      "[svelte-i18n] Cannot format a message without first setting the initial locale."
    );
  let m = ss(e, d);
  if (!m)
    m = (o = (s = (n = (r = Ur()).handleMissingMessage) == null ? void 0 : n.call(r, { locale: d, id: e, defaultValue: h })) != null ? s : h) != null ? o : e;
  else if (typeof m != "string")
    return console.warn(
      `[svelte-i18n] Message with id "${e}" must be of type "string", found: "${typeof m}". Gettin its value through the "$format" method is deprecated; use the "json" method instead.`
    ), m;
  if (!c)
    return m;
  let T = m;
  try {
    T = $u(m, d).format(c);
  } catch (w) {
    w instanceof Error && console.warn(
      `[svelte-i18n] Message "${e}" has syntax error:`,
      w.message
    );
  }
  return T;
}, tf = (e, t) => Qu(t).format(e), rf = (e, t) => Ku(t).format(e), nf = (e, t) => Ju(t).format(e), af = (e, t = Hr()) => ss(e, t);
Xr([Wr, sn], () => ef);
Xr([Wr], () => tf);
Xr([Wr], () => rf);
Xr([Wr], () => nf);
Xr([Wr, sn], () => af);
const sf = "__i18n__", of = [
  "label",
  "info",
  "placeholder",
  "description",
  "title",
  "value"
], lf = [
  "elem_id",
  "elem_classes",
  "visible",
  "interactive",
  "server_fns",
  "server",
  "id",
  "target",
  "theme_mode",
  "version",
  "root",
  "autoscroll",
  "max_file_size",
  "formatter",
  "client",
  "load_component",
  "scale",
  "min_width",
  "theme",
  "padding",
  "loading_status",
  "label",
  "show_label",
  "validation_error",
  "show_progress",
  "api_prefix",
  "container",
  "attached_events",
  "register_component",
  "dispatcher"
];
function uf(e) {
  return typeof e == "string" && e.includes(sf);
}
class hs {
  load_component;
  #t = q(Kt({}));
  get shared() {
    return i(this.#t);
  }
  set shared(t) {
    g(this.#t, t, !0);
  }
  #r = q(Kt({}));
  get props() {
    return i(this.#r);
  }
  set props(t) {
    g(this.#r, t, !0);
  }
  #e = q((t) => t);
  get i18n() {
    return i(this.#e);
  }
  set i18n(t) {
    g(this.#e, t, !0);
  }
  translatable_props = {};
  dispatcher;
  last_update = null;
  shared_props = lf;
  mounted = !1;
  old_value;
  register_component;
  constructor(t, r) {
    for (const n in t.shared_props)
      this.shared[n] = t.shared_props[n];
    for (const n in t.props)
      this.props[n] = t.props[n];
    if (r)
      for (const n in r)
        this.props[n] === void 0 && (this.props[n] = r[n]);
    this.i18n = this.props.i18n ?? ((n) => n);
    for (const n of of)
      this.shared[n] = this._translate_and_store(
        "shared",
        n,
        // @ts-ignore
        t.shared_props[n]
      ), this.props[n] = this._translate_and_store(
        "props",
        n,
        // @ts-ignore
        t.props[n]
      );
    this.load_component = this.shared.load_component, this.register_component = this.shared.register_component || (() => {
    }), this.dispatcher = this.shared.dispatcher || (() => {
    }), this.register_component(
      t.shared_props.id,
      // @ts-ignore
      this.set_data.bind(this),
      this.get_data.bind(this)
    ), Nt(() => {
      for (const n in t.shared_props)
        this._is_i18n_managed(`shared.${n}`, t.shared_props[n]) || (this.shared[n] = t.shared_props[n]);
      for (const n in t.props)
        this._is_i18n_managed(`props.${n}`, t.props[n]) || (this.props[n] = t.props[n]);
      this.register_component(
        t.shared_props.id,
        // @ts-ignore
        this.set_data.bind(this),
        this.get_data.bind(this)
      ), $e(() => {
        this.shared.id = t.shared_props.id;
      });
    }), Object.keys(this.translatable_props).length > 0 && Wr.subscribe(() => {
      for (const [n, s] of Object.entries(this.translatable_props)) {
        const [o, f] = n.split("."), c = this.i18n(s);
        o === "shared" ? this.shared[f] = c : this.props[f] = c;
      }
    });
  }
  // check if props are translatable
  _is_i18n_managed(t, r) {
    const n = this.translatable_props[t];
    return n ? r === n ? !0 : (delete this.translatable_props[t], !1) : !1;
  }
  _translate_and_store(t, r, n) {
    if (typeof n != "string") return n;
    const s = this.i18n(n);
    return s !== n && (this.translatable_props[`${t}.${r}`] = n), s;
  }
  dispatch(t, r) {
    this.dispatcher(this.shared.id, t, r);
  }
  async get_data() {
    return Yo(this.props);
  }
  update(t) {
    this.set_data(t);
  }
  set_data(t) {
    for (const r in t) {
      const n = t[r], s = uf(n) ? this._translate_and_store(this.shared_props.includes(r) ? "shared" : "props", r, n) : n;
      if (this.shared_props.includes(r)) {
        const o = r;
        this.shared[o] = s;
        continue;
      }
      this.props[r] = s;
    }
  }
  watch_for_change() {
    Nt(() => {
      this.mounted || (this.old_value = this.props.value, this.mounted = !0), this.old_value != this.props.value && (this.old_value = this.props.value, this.dispatch("change"));
    });
  }
}
Vo();
var ff = /* @__PURE__ */ ka('<svg class="resize-handle svelte-1stq1b1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><line x1="1" y1="9" x2="9" y2="1" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line><line x1="5" y1="9" x2="9" y2="5" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line></svg>'), ma = /* @__PURE__ */ xe("<!> <!>", 1), cf = /* @__PURE__ */ xe('<div class="placeholder svelte-1stq1b1"></div>');
function ds(e, t) {
  Vr(t, !1);
  let r = X(t, "height", 8, void 0), n = X(t, "min_height", 8, void 0), s = X(t, "max_height", 8, void 0), o = X(t, "width", 8, void 0), f = X(t, "elem_id", 8, ""), c = X(t, "elem_classes", 24, () => []), d = X(t, "variant", 8, "solid"), h = X(t, "border_mode", 8, "base"), m = X(t, "padding", 8, !0), T = X(t, "type", 8, "normal"), w = X(t, "test_id", 8, void 0), L = X(t, "explicit_call", 8, !1), R = X(t, "container", 8, !0), P = X(t, "visible", 8, !0), H = X(t, "allow_overflow", 8, !0), _ = X(t, "overflow_behavior", 8, "auto"), y = X(t, "scale", 8, null), S = X(t, "min_width", 8, 0), E = X(t, "flex", 12, !1), b = X(t, "resizable", 8, !1), A = X(t, "rtl", 8, !1), N = X(t, "fullscreen", 12, !1), W = X(t, "label", 8, void 0), Y = Sr(N()), V = Sr(), Ce = T() === "fieldset" ? "fieldset" : "div", Ge = Sr(0), De = Sr(0), Me = Sr(null);
  function Ie(de) {
    N() && de.key === "Escape" && N(!1);
  }
  const Fe = (de) => {
    if (de !== void 0) {
      if (typeof de == "number")
        return de + "px";
      if (typeof de == "string")
        return de;
    }
  }, Xe = (de) => {
    let re = de.clientY;
    const Ve = (pe) => {
      const Ne = pe.clientY - re;
      re = pe.clientY, Xo(V, i(V).style.height = `${i(V).offsetHeight + Ne}px`);
    }, Le = () => {
      window.removeEventListener("mousemove", Ve), window.removeEventListener("mouseup", Le);
    };
    window.addEventListener("mousemove", Ve), window.addEventListener("mouseup", Le);
  };
  Yi(
    () => (kt(N()), i(Y), i(V)),
    () => {
      N() !== i(Y) && (g(Y, N()), N() ? (g(Me, i(V).getBoundingClientRect()), g(Ge, i(V).offsetHeight), g(De, i(V).offsetWidth), window.addEventListener("keydown", Ie)) : (g(Me, null), window.removeEventListener("keydown", Ie)));
    }
  ), Yi(() => kt(P()), () => {
    P() || E(!1);
  }), zo(), Ml();
  var Ue = Tr(), it = ft(Ue);
  {
    var ke = (de) => {
      var re = ma(), Ve = ft(re);
      hl(Ve, () => Ce, !1, (Ne, ct) => {
        Nn(Ne, (tt) => g(V, tt), () => i(V)), Tl(
          Ne,
          (tt, st) => ({
            "data-testid": w(),
            id: f(),
            class: `block ${tt ?? ""}`,
            dir: A() ? "rtl" : "ltr",
            "aria-label": W(),
            style: "",
            [en]: {
              hidden: P() === "hidden",
              padded: m(),
              flex: E(),
              border_focus: h() === "focus",
              border_contrast: h() === "contrast",
              "hide-container": !L() && !R(),
              fullscreen: N(),
              animating: N() && i(Me) !== null,
              "auto-margin": y() === null
            },
            [kr]: st
          }),
          [
            () => (kt(c()), $e(() => c()?.join(" ") || "")),
            () => ({
              height: (kt(N()), kt(r()), $e(() => N() ? void 0 : Fe(r()))),
              "min-height": (kt(N()), kt(n()), $e(() => N() ? void 0 : Fe(n()))),
              "max-height": (kt(N()), kt(s()), $e(() => N() ? void 0 : Fe(s()))),
              "--start-top": (i(Me), $e(() => i(Me) ? `${i(Me).top}px` : "0px")),
              "--start-left": (i(Me), $e(() => i(Me) ? `${i(Me).left}px` : "0px")),
              "--start-width": (i(Me), $e(() => i(Me) ? `${i(Me).width}px` : "0px")),
              "--start-height": (i(Me), $e(() => i(Me) ? `${i(Me).height}px` : "0px")),
              width: (kt(N()), kt(o()), $e(() => N() ? void 0 : typeof o() == "number" ? `calc(min(${o()}px, 100%))` : Fe(o()))),
              "border-style": d(),
              overflow: H() ? _() : "hidden",
              "flex-grow": y(),
              "min-width": `calc(min(${S()}px, 100%))`
            })
          ],
          void 0,
          void 0,
          "svelte-1stq1b1"
        );
        var at = ma(), wt = ft(at);
        ri(wt, t, "default", {});
        var et = U(wt, 2);
        {
          var It = (tt) => {
            var st = ff();
            te("mousedown", st, Xe), Z(tt, st);
          };
          _e(et, (tt) => {
            b() && tt(It);
          });
        }
        Z(ct, at);
      });
      var Le = U(Ve, 2);
      {
        var pe = (Ne) => {
          var ct = cf();
          let at;
          be(() => at = Ht(ct, "", at, {
            height: i(Ge) + "px",
            width: i(De) + "px"
          })), Z(Ne, ct);
        };
        _e(Le, (Ne) => {
          N() && Ne(pe);
        });
      }
      Z(de, re);
    };
    _e(it, (de) => {
      (P() === !0 || P() === "hidden") && de(ke);
    });
  }
  Z(e, Ue), zr();
}
var hf = /* @__PURE__ */ xe('<span class="svelte-vvirtv"> </span>'), df = /* @__PURE__ */ xe("<button><!> <div><!> <!></div></button>");
function _a(e, t) {
  let r = X(t, "label", 3, ""), n = X(t, "show_label", 3, !1), s = X(t, "pending", 3, !1), o = X(t, "size", 3, "small"), f = X(t, "padded", 3, !0), c = X(t, "highlight", 3, !1), d = X(t, "disabled", 3, !1), h = X(t, "hasPopup", 3, !1), m = X(t, "color", 3, "var(--block-label-text-color)"), T = X(t, "transparent", 3, !1), w = X(t, "background", 3, "var(--block-background-fill)"), L = X(t, "border", 3, "transparent"), R = Ke(() => c() ? "var(--color-accent)" : m());
  var P = df();
  let H, _;
  var y = J(P);
  {
    var S = (Y) => {
      var V = hf(), Ce = J(V);
      be(() => ye(Ce, r())), Z(Y, V);
    };
    _e(y, (Y) => {
      n() && Y(S);
    });
  }
  var E = U(y, 2);
  let b;
  var A = J(E);
  ul(A, () => t.Icon, (Y, V) => {
    V(Y, {});
  });
  var N = U(A, 2);
  {
    var W = (Y) => {
      var V = Tr(), Ce = ft(V);
      nl(Ce, () => t.children), Z(Y, V);
    };
    _e(N, (Y) => {
      t.children && Y(W);
    });
  }
  be(() => {
    H = Ot(P, 1, "icon-button svelte-vvirtv", null, H, {
      pending: s(),
      padded: f(),
      highlight: c(),
      transparent: T()
    }), P.disabled = d(), mr(P, "aria-label", r()), mr(P, "aria-haspopup", h()), mr(P, "title", r()), _ = Ht(P, "", _, {
      "--border-color": L(),
      color: !d() && i(R) ? i(R) : "var(--block-label-text-color)",
      "--bg-color": d() ? "auto" : w()
    }), b = Ot(E, 1, "svelte-vvirtv", null, b, {
      "x-small": o() === "x-small",
      small: o() === "small",
      large: o() === "large",
      medium: o() === "medium"
    });
  }), Ca("click", P, function(...Y) {
    t.onclick?.apply(this, Y);
  }), Z(e, P);
}
An(["click"]);
var pf = /* @__PURE__ */ ka('<svg width="100%" height="100%" viewBox="0 0 24 24" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:space="preserve" stroke="currentColor" style="fill-rule:evenodd;clip-rule:evenodd;stroke-linecap:round;stroke-linejoin:round;"><g transform="matrix(1.14096,-0.140958,-0.140958,1.14096,-0.0559523,0.0559523)"><path d="M18,6L6.087,17.913" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></g><path d="M4.364,4.364L19.636,19.636" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></svg>');
function ba(e) {
  var t = pf();
  Z(e, t);
}
An(["click"]);
function Zn(e) {
  let t = ["", "k", "M", "G", "T", "P", "E", "Z"], r = 0;
  for (; e > 1e3 && r < t.length - 1; )
    e /= 1e3, r++;
  let n = t[r];
  return (Number.isInteger(e) ? e : e.toFixed(1)) + n;
}
function ya(e) {
  return Object.prototype.toString.call(e) === "[object Date]";
}
function di(e, t, r, n) {
  if (typeof r == "number" || ya(r)) {
    const s = n - r, o = (r - t) / (e.dt || 1 / 60), f = e.opts.stiffness * s, c = e.opts.damping * o, d = (f - c) * e.inv_mass, h = (o + d) * e.dt;
    return Math.abs(h) < e.opts.precision && Math.abs(s) < e.opts.precision ? n : (e.settled = !1, ya(r) ? new Date(r.getTime() + h) : r + h);
  } else {
    if (Array.isArray(r))
      return r.map(
        (s, o) => (
          // @ts-ignore
          di(e, t[o], r[o], n[o])
        )
      );
    if (typeof r == "object") {
      const s = {};
      for (const o in r)
        s[o] = di(e, t[o], r[o], n[o]);
      return s;
    } else
      throw new Error(`Cannot spring ${typeof r} values`);
  }
}
function xa(e, t = {}) {
  const r = an(e), { stiffness: n = 0.15, damping: s = 0.8, precision: o = 0.01 } = t;
  let f, c, d, h = (
    /** @type {T} */
    e
  ), m = (
    /** @type {T | undefined} */
    e
  ), T = 1, w = 0, L = !1;
  function R(H, _ = {}) {
    m = H;
    const y = d = {};
    return e == null || _.hard || P.stiffness >= 1 && P.damping >= 1 ? (L = !0, f = Wt.now(), h = H, r.set(e = m), Promise.resolve()) : (_.soft && (w = 1 / ((_.soft === !0 ? 0.5 : +_.soft) * 60), T = 0), c || (f = Wt.now(), L = !1, c = cl((S) => {
      if (L)
        return L = !1, c = null, !1;
      T = Math.min(T + w, 1);
      const E = Math.min(S - f, 1e3 / 30), b = {
        inv_mass: T,
        opts: P,
        settled: !0,
        dt: E * 60 / 1e3
      }, A = di(b, h, e, m);
      return f = S, h = /** @type {T} */
      e, r.set(e = /** @type {T} */
      A), b.settled && (c = null), !b.settled;
    })), new Promise((S) => {
      c.promise.then(() => {
        y === d && S();
      });
    }));
  }
  const P = {
    set: R,
    update: (H, _) => R(H(
      /** @type {T} */
      m,
      /** @type {T} */
      e
    ), _),
    subscribe: r.subscribe,
    stiffness: n,
    damping: s,
    precision: o
  };
  return P;
}
var vf = /* @__PURE__ */ xe('<div><svg viewBox="-1200 -1200 3000 3000" fill="none" xmlns="http://www.w3.org/2000/svg" class="svelte-m6d381"><g><path d="M255.926 0.754768L509.702 139.936V221.027L255.926 81.8465V0.754768Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M509.69 139.936L254.981 279.641V361.255L509.69 221.55V139.936Z" fill="#FF7C00" class="svelte-m6d381"></path><path d="M0.250138 139.937L254.981 279.641V361.255L0.250138 221.55V139.937Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M255.923 0.232622L0.236328 139.936V221.55L255.923 81.8469V0.232622Z" fill="#FF7C00" class="svelte-m6d381"></path></g><g><path d="M255.926 141.5L509.702 280.681V361.773L255.926 222.592V141.5Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M509.69 280.679L254.981 420.384V501.998L509.69 362.293V280.679Z" fill="#FF7C00" class="svelte-m6d381"></path><path d="M0.250138 280.681L254.981 420.386V502L0.250138 362.295V280.681Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M255.923 140.977L0.236328 280.68V362.294L255.923 222.591V140.977Z" fill="#FF7C00" class="svelte-m6d381"></path></g></svg></div>');
function gf(e, t) {
  Vr(t, !0);
  const r = () => Ji(d, "$top", s), n = () => Ji(h, "$bottom", s), [s, o] = Ko();
  var f = this && this.__awaiter || function(S, E, b, A) {
    function N(W) {
      return W instanceof b ? W : new b(function(Y) {
        Y(W);
      });
    }
    return new (b || (b = Promise))(function(W, Y) {
      function V(De) {
        try {
          Ge(A.next(De));
        } catch (Me) {
          Y(Me);
        }
      }
      function Ce(De) {
        try {
          Ge(A.throw(De));
        } catch (Me) {
          Y(Me);
        }
      }
      function Ge(De) {
        De.done ? W(De.value) : N(De.value).then(V, Ce);
      }
      Ge((A = A.apply(S, E || [])).next());
    });
  };
  let c = X(t, "margin", 3, !0);
  const d = xa([0, 0]), h = xa([0, 0]);
  let m = q(!1);
  function T() {
    return f(this, void 0, void 0, function* () {
      yield Promise.all([d.set([125, 140]), h.set([-125, -140])]), yield Promise.all([d.set([-125, 140]), h.set([125, -140])]), yield Promise.all([d.set([-125, 0]), h.set([125, -0])]), yield Promise.all([d.set([125, 0]), h.set([-125, 0])]);
    });
  }
  function w() {
    return f(this, void 0, void 0, function* () {
      yield T(), i(m) || w();
    });
  }
  function L() {
    return f(this, void 0, void 0, function* () {
      yield Promise.all([d.set([125, 0]), h.set([-125, 0])]), w();
    });
  }
  Nt(() => (L(), () => {
    g(m, !0);
  }));
  var R = vf();
  let P;
  var H = J(R), _ = J(H), y = U(_);
  be(() => {
    P = Ot(R, 1, "svelte-m6d381", null, P, { margin: c() }), Ht(_, `transform: translate(${r()[0] ?? ""}px, ${r()[1] ?? ""}px);`), Ht(y, `transform: translate(${n()[0] ?? ""}px, ${n()[1] ?? ""}px);`);
  }), Z(e, R), zr(), o();
}
var mf = function(e, t, r, n) {
  function s(o) {
    return o instanceof r ? o : new r(function(f) {
      f(o);
    });
  }
  return new (r || (r = Promise))(function(o, f) {
    function c(m) {
      try {
        h(n.next(m));
      } catch (T) {
        f(T);
      }
    }
    function d(m) {
      try {
        h(n.throw(m));
      } catch (T) {
        f(T);
      }
    }
    function h(m) {
      m.done ? o(m.value) : s(m.value).then(c, d);
    }
    h((n = n.apply(e, t || [])).next());
  });
};
let gn = [], Jn = !1;
const _f = typeof window < "u", ps = _f ? window.requestAnimationFrame : (e) => {
};
function bf(e) {
  return mf(this, arguments, void 0, function* (t, r = !0) {
    if (!(window.__gradio_mode__ === "website" || window.__gradio_mode__ !== "app" && r !== !0)) {
      if (gn.push(t), !Jn) Jn = !0;
      else return;
      yield Wo(), ps(() => {
        let n = [0, 0];
        for (let s = 0; s < gn.length; s++) {
          const f = gn[s].getBoundingClientRect();
          (s === 0 || f.top + window.scrollY <= n[0]) && (n[0] = f.top + window.scrollY, n[1] = s);
        }
        window.scrollTo({ top: n[0] - 20, behavior: "smooth" }), Jn = !1, gn = [];
      });
    }
  });
}
var yf = /* @__PURE__ */ xe('<div class="validation-error svelte-124hqw6"> <button class="svelte-124hqw6"><!></button></div>'), xf = /* @__PURE__ */ xe('<div class="eta-bar svelte-124hqw6"></div>'), wf = /* @__PURE__ */ xe("<!> ", 1), Ef = /* @__PURE__ */ xe("<!> <!> <!> <!>", 1), Sf = /* @__PURE__ */ xe('<div class="progress-level svelte-124hqw6"><div class="progress-level-inner svelte-124hqw6"><!></div> <div class="progress-bar-wrap svelte-124hqw6"><div class="progress-bar svelte-124hqw6"></div></div></div>'), Tf = /* @__PURE__ */ xe('<p class="loading svelte-124hqw6"> </p> <!>', 1), Mf = /* @__PURE__ */ xe("<!> <div><!> <!></div> <!> <!>", 1), Af = /* @__PURE__ */ xe('<div class="clear-status svelte-124hqw6"><!></div> <span class="error svelte-124hqw6"> </span> <!>', 1), Of = /* @__PURE__ */ xe("<div> <!> </div>"), Nf = /* @__PURE__ */ xe('<div data-testid="status-tracker"><!> <!></div> <!>', 1);
function Hf(e, t) {
  Vr(t, !0);
  let r = X(t, "eta", 3, null), n = X(t, "scroll_to_output", 3, !1), s = X(t, "timer", 3, !0), o = X(t, "show_progress", 3, "full"), f = X(t, "message", 3, null), c = X(t, "progress", 3, null), d = X(t, "variant", 3, "default"), h = X(t, "loading_text", 3, "Loading..."), m = X(t, "absolute", 3, !0), T = X(t, "translucent", 3, !1), w = X(t, "border", 3, !1), L = X(t, "validation_error", 7, null), R = X(t, "show_validation_error", 3, !0), P = X(t, "type", 3, null), H = X(t, "used_cache", 3, null), _ = X(t, "cache_duration", 3, null), y = X(t, "avg_time", 3, null), S, E = !1, b = q(0), A = q(null), N = q(null), W = q(!1), Y = q(null), V = q(!1), Ce = q(!1), Ge = q(null), De = q(null), Me = q("from cache"), Ie = q(!1), Fe = null, Xe = null;
  const Ue = Ke(() => !(R() && L()) && (P() === "input" || !t.status || t.status === "complete" || o() === "hidden" || t.status == "streaming"));
  let it = q(0);
  const ke = Ke(() => i(N) === null || i(N) <= 0 || !i(it) ? 0 : Math.min(i(it) / i(N), 1)), de = Ke(() => i(it).toFixed(1));
  let re = Ke(() => c() == null), Ve = Ke(() => r() !== null && r() !== void 0 ? r() : i(A));
  function Le() {
    ps(() => {
      g(it, (performance.now() - i(b)) / 1e3), E && Le();
    });
  }
  let pe = Ke(() => {
    let fe = null;
    c() != null ? fe = c().map((He) => {
      if (He.index != null && He.length != null)
        return He.index / He.length;
      if (He.progress != null)
        return He.progress;
    }) : fe = null;
    let we, Be = "";
    return fe ? (we = fe[fe.length - 1], we === 0 ? Be = "0" : Be = "150ms") : we = void 0, {
      progress_level: fe,
      last_progress_level: we,
      progress_bar_transition: Be
    };
  });
  function Ne() {
    E || (g(A, g(Y, null), !0), g(b, performance.now(), !0), E = !0, Le());
  }
  function ct() {
    g(A, g(Y, null), !0), E && (E = !1);
  }
  Nt(() => {
    t.status === "pending" ? Ne() : $e(() => {
      ct();
    });
  }), Nt(() => {
    S && n() && (t.status === "pending" || t.status === "complete") && bf(S, t.autoscroll);
  }), Nt(() => {
    i(Ve) != null && i(A) !== i(Ve) && (g(N, (performance.now() - i(b)) / 1e3 + i(Ve)), g(Y, i(N).toFixed(1), !0), g(A, i(Ve), !0));
  });
  function at() {
    g(W, !1);
  }
  Nt(() => {
    $e(() => {
      at();
    }), t.status === "error" && f() && g(W, !0);
  }), Nt(() => {
    t.status === "complete" && P() === "output" && H() && _() != null && (g(Ge, _().toFixed(1), !0), g(Me, H() === "full" ? "from cache" : "used cache", !0), g(Ie, y() != null && y() > _() && y() > 0, !0), g(De, i(Ie) ? y().toFixed(1) : null, !0), g(V, !0), g(Ce, !1), Fe && clearTimeout(Fe), Xe && clearTimeout(Xe), Fe = setTimeout(
      () => {
        g(Ce, !0), Xe = setTimeout(
          () => {
            g(V, !1), g(Ce, !1);
          },
          500
        );
      },
      1750
    ));
  });
  var wt = Nf(), et = ft(wt);
  let It, tt;
  var st = J(et);
  {
    var Ye = (fe) => {
      var we = yf(), Be = J(we), He = U(Be), Ee = J(He);
      {
        let ht = Ke(() => t.i18n ? t.i18n("common.clear") : "Clear");
        _a(Ee, {
          get Icon() {
            return ba;
          },
          get label() {
            return i(ht);
          },
          disabled: !1,
          size: "x-small",
          background: "var(--background-fill-primary)",
          color: "var(--error-background-text)",
          border: "var(--border-color-primary)",
          onclick: () => L(null)
        });
      }
      be(() => ye(Be, `${L() ?? ""} `)), Z(fe, we);
    };
    _e(st, (fe) => {
      L() && R() && fe(Ye);
    });
  }
  var Pt = U(st, 2);
  {
    var ae = (fe) => {
      var we = Mf(), Be = ft(we);
      {
        var He = (Q) => {
          var Se = xf();
          let dt;
          be(() => dt = Ht(Se, "", dt, {
            transform: `translateX(${(i(ke) || 0) * 100 - 100}%)`
          })), Z(Q, Se);
        };
        _e(Be, (Q) => {
          d() === "default" && i(re) && o() === "full" && Q(He);
        });
      }
      var Ee = U(Be, 2);
      let ht;
      var Ct = J(Ee);
      {
        var Pe = (Q) => {
          var Se = Tr(), dt = ft(Se);
          Ar(dt, 17, c, Mr, (Et, pt) => {
            var ve = Tr(), Te = ft(ve);
            {
              var qt = (Ft) => {
                var tr = wf(), Yt = ft(tr);
                {
                  var rr = (St) => {
                    var lt = Jt();
                    be((bt, nr) => ye(lt, `${bt ?? ""}/${nr ?? ""}`), [
                      () => Zn(i(pt).index || 0),
                      () => Zn(i(pt).length)
                    ]), Z(St, lt);
                  }, jt = (St) => {
                    var lt = Jt();
                    be((bt) => ye(lt, bt), [() => Zn(i(pt).index || 0)]), Z(St, lt);
                  };
                  _e(Yt, (St) => {
                    i(pt).length != null ? St(rr) : St(jt, -1);
                  });
                }
                var Ut = U(Yt);
                be(() => ye(Ut, ` ${i(pt).unit ?? ""} |  `)), Z(Ft, tr);
              };
              _e(Te, (Ft) => {
                i(pt).index != null && Ft(qt);
              });
            }
            Z(Et, ve);
          }), Z(Q, Se);
        }, ce = (Q) => {
          var Se = Jt();
          be(() => ye(Se, `queue: ${t.queue_position + 1}/${t.queue_size ?? ""} |`)), Z(Q, Se);
        }, $t = (Q) => {
          var Se = Jt("processing |");
          Z(Q, Se);
        };
        _e(Ct, (Q) => {
          c() ? Q(Pe) : t.queue_position !== null && t.queue_size !== void 0 && t.queue_position >= 0 ? Q(ce, 1) : t.queue_position === 0 && Q($t, 2);
        });
      }
      var Qe = U(Ct, 2);
      {
        var er = (Q) => {
          var Se = Jt();
          be(() => ye(Se, `${i(de) ?? ""}${r() ? `/${i(Y)}` : ""}s`)), Z(Q, Se);
        };
        _e(Qe, (Q) => {
          s() && Q(er);
        });
      }
      var ie = U(Ee, 2);
      {
        var _t = (Q) => {
          var Se = Sf(), dt = J(Se), Et = J(dt);
          {
            var pt = (Ft) => {
              var tr = Tr(), Yt = ft(tr);
              Ar(Yt, 17, c, Mr, (rr, jt, Ut) => {
                var St = Tr(), lt = ft(St);
                {
                  var bt = (nr) => {
                    var We = Ef(), u = ft(We);
                    {
                      var v = (j) => {
                        var ee = Jt(" /");
                        Z(j, ee);
                      };
                      _e(u, (j) => {
                        Ut !== 0 && j(v);
                      });
                    }
                    var O = U(u, 2);
                    {
                      var C = (j) => {
                        var ee = Jt();
                        be(() => ye(ee, i(jt).desc)), Z(j, ee);
                      };
                      _e(O, (j) => {
                        i(jt).desc != null && j(C);
                      });
                    }
                    var G = U(O, 2);
                    {
                      var k = (j) => {
                        var ee = Jt("-");
                        Z(j, ee);
                      };
                      _e(G, (j) => {
                        i(jt).desc != null && i(pe).progress_level && i(pe).progress_level[Ut] != null && j(k);
                      });
                    }
                    var z = U(G, 2);
                    {
                      var K = (j) => {
                        var ee = Jt();
                        be((oe) => ye(ee, `${oe ?? ""}%`), [
                          () => (100 * (i(pe).progress_level[Ut] || 0)).toFixed(1)
                        ]), Z(j, ee);
                      };
                      _e(z, (j) => {
                        i(pe).progress_level != null && j(K);
                      });
                    }
                    Z(nr, We);
                  };
                  _e(lt, (nr) => {
                    (i(jt).desc != null || i(pe).progress_level && i(pe).progress_level[Ut] != null) && nr(bt);
                  });
                }
                Z(rr, St);
              }), Z(Ft, tr);
            };
            _e(Et, (Ft) => {
              c() != null && Ft(pt);
            });
          }
          var ve = U(dt, 2), Te = J(ve);
          let qt;
          be(() => qt = Ht(Te, "", qt, {
            width: `${i(pe).last_progress_level * 100}%`,
            transition: i(pe).progress_bar_transition
          })), Z(Q, Se);
        }, Gt = (Q) => {
          {
            let Se = Ke(() => d() === "default");
            gf(Q, {
              get margin() {
                return i(Se);
              }
            });
          }
        };
        _e(ie, (Q) => {
          i(pe).last_progress_level != null ? Q(_t) : o() === "full" && Q(Gt, 1);
        });
      }
      var se = U(ie, 2);
      {
        var Dt = (Q) => {
          var Se = Tf(), dt = ft(Se), Et = J(dt), pt = U(dt, 2);
          ri(pt, t, "additional-loading-text", {}), be(() => ye(Et, h())), Z(Q, Se);
        };
        _e(se, (Q) => {
          s() || Q(Dt);
        });
      }
      be(() => ht = Ot(Ee, 1, "progress-text svelte-124hqw6", null, ht, {
        "meta-text-center": d() === "center",
        "meta-text": d() === "default"
      })), Z(fe, we);
    }, gt = (fe) => {
      var we = Af(), Be = ft(we), He = J(Be);
      {
        let Pe = Ke(() => t.i18n("common.clear"));
        _a(He, {
          get Icon() {
            return ba;
          },
          get label() {
            return i(Pe);
          },
          disabled: !1,
          $$events: {
            click: () => {
              t.on_clear_status?.();
            }
          }
        });
      }
      var Ee = U(Be, 2), ht = J(Ee), Ct = U(Ee, 2);
      ri(Ct, t, "error", {}), be((Pe) => ye(ht, Pe), [() => t.i18n("common.error")]), Z(fe, we);
    };
    _e(Pt, (fe) => {
      t.status === "pending" ? fe(ae) : t.status === "error" && fe(gt, 1);
    });
  }
  Nn(et, (fe) => S = fe, () => S);
  var ot = U(et, 2);
  {
    var mt = (fe) => {
      var we = Of();
      let Be, He;
      var Ee = J(we), ht = U(Ee);
      {
        var Ct = (ce) => {
          var $t = Jt();
          be(() => ye($t, `~${i(De) ?? ""}s
			→ `)), Z(ce, $t);
        };
        _e(ht, (ce) => {
          i(Ie) && ce(Ct);
        });
      }
      var Pe = U(ht);
      be(() => {
        Be = Ot(we, 1, "cache-indicator svelte-124hqw6", null, Be, { "fade-out": i(Ce) }), He = Ht(we, "", He, { position: m() ? "absolute" : "static" }), ye(Ee, `⚡ ${i(Me) ?? ""}: `), ye(Pe, `${i(Ge) ?? ""}s`);
      }), Z(fe, we);
    };
    _e(ot, (fe) => {
      i(V) && fe(mt);
    });
  }
  be(() => {
    It = Ot(et, 1, `wrap ${d() ?? ""} ${o() ?? ""}`, "svelte-124hqw6", It, {
      "no-click": L() && R(),
      hide: i(Ue),
      translucent: d() === "center" && (t.status === "pending" || t.status === "error") || T() || o() === "minimal" || L(),
      generating: t.status === "generating" && o() === "full",
      border: w()
    }), tt = Ht(et, "", tt, {
      position: m() ? "absolute" : "static",
      padding: m() ? "0" : "var(--size-8) 0"
    });
  }), Z(e, wt), zr();
}
const If = (e) => {
  const t = {};
  for (let r = 0, n = e.length; r < n; r++) {
    const s = e[r];
    for (const o in s)
      t[o] ? t[o] = t[o].concat(s[o]) : t[o] = s[o];
  }
  return t;
}, Pf = [
  "abbr",
  "accept",
  "accept-charset",
  "accesskey",
  "action",
  "align",
  "alink",
  "allow",
  "allowfullscreen",
  "alt",
  "anchor",
  "archive",
  "as",
  "async",
  "autocapitalize",
  "autocomplete",
  "autocorrect",
  "autofocus",
  "autopictureinpicture",
  "autoplay",
  "axis",
  "background",
  "behavior",
  "bgcolor",
  "border",
  "bordercolor",
  "capture",
  "cellpadding",
  "cellspacing",
  "challenge",
  "char",
  "charoff",
  "charset",
  "checked",
  "cite",
  "class",
  "classid",
  "clear",
  "code",
  "codebase",
  "codetype",
  "color",
  "cols",
  "colspan",
  "compact",
  "content",
  "contenteditable",
  "controls",
  "controlslist",
  "conversiondestination",
  "coords",
  "crossorigin",
  "csp",
  "data",
  "datetime",
  "declare",
  "decoding",
  "default",
  "defer",
  "dir",
  "direction",
  "dirname",
  "disabled",
  "disablepictureinpicture",
  "disableremoteplayback",
  "disallowdocumentaccess",
  "download",
  "draggable",
  "elementtiming",
  "enctype",
  "end",
  "enterkeyhint",
  "event",
  "exportparts",
  "face",
  "for",
  "form",
  "formaction",
  "formenctype",
  "formmethod",
  "formnovalidate",
  "formtarget",
  "frame",
  "frameborder",
  "headers",
  "height",
  "hidden",
  "high",
  "href",
  "hreflang",
  "hreftranslate",
  "hspace",
  "http-equiv",
  "id",
  "imagesizes",
  "imagesrcset",
  "importance",
  "impressiondata",
  "impressionexpiry",
  "incremental",
  "inert",
  "inputmode",
  "integrity",
  "invisible",
  "ismap",
  "keytype",
  "kind",
  "label",
  "lang",
  "language",
  "latencyhint",
  "leftmargin",
  "link",
  "list",
  "loading",
  "longdesc",
  "loop",
  "low",
  "lowsrc",
  "manifest",
  "marginheight",
  "marginwidth",
  "max",
  "maxlength",
  "mayscript",
  "media",
  "method",
  "min",
  "minlength",
  "multiple",
  "muted",
  "name",
  "nohref",
  "nomodule",
  "nonce",
  "noresize",
  "noshade",
  "novalidate",
  "nowrap",
  "object",
  "open",
  "optimum",
  "part",
  "pattern",
  "ping",
  "placeholder",
  "playsinline",
  "policy",
  "poster",
  "preload",
  "pseudo",
  "readonly",
  "referrerpolicy",
  "rel",
  "reportingorigin",
  "required",
  "resources",
  "rev",
  "reversed",
  "role",
  "rows",
  "rowspan",
  "rules",
  "sandbox",
  "scheme",
  "scope",
  "scopes",
  "scrollamount",
  "scrolldelay",
  "scrolling",
  "select",
  "selected",
  "shadowroot",
  "shadowrootdelegatesfocus",
  "shape",
  "size",
  "sizes",
  "slot",
  "span",
  "spellcheck",
  "src",
  "srclang",
  "srcset",
  "standby",
  "start",
  "step",
  "style",
  "summary",
  "tabindex",
  "target",
  "text",
  "title",
  "topmargin",
  "translate",
  "truespeed",
  "trusttoken",
  "type",
  "usemap",
  "valign",
  "value",
  "valuetype",
  "version",
  "virtualkeyboardpolicy",
  "vlink",
  "vspace",
  "webkitdirectory",
  "width",
  "wrap"
], Cf = [
  "accent-height",
  "accumulate",
  "additive",
  "alignment-baseline",
  "ascent",
  "attributename",
  "attributetype",
  "azimuth",
  "basefrequency",
  "baseline-shift",
  "begin",
  "bias",
  "by",
  "class",
  "clip",
  "clippathunits",
  "clip-path",
  "clip-rule",
  "color",
  "color-interpolation",
  "color-interpolation-filters",
  "color-profile",
  "color-rendering",
  "cx",
  "cy",
  "d",
  "dx",
  "dy",
  "diffuseconstant",
  "direction",
  "display",
  "divisor",
  "dominant-baseline",
  "dur",
  "edgemode",
  "elevation",
  "end",
  "fill",
  "fill-opacity",
  "fill-rule",
  "filter",
  "filterunits",
  "flood-color",
  "flood-opacity",
  "font-family",
  "font-size",
  "font-size-adjust",
  "font-stretch",
  "font-style",
  "font-variant",
  "font-weight",
  "fx",
  "fy",
  "g1",
  "g2",
  "glyph-name",
  "glyphref",
  "gradientunits",
  "gradienttransform",
  "height",
  "href",
  "id",
  "image-rendering",
  "in",
  "in2",
  "k",
  "k1",
  "k2",
  "k3",
  "k4",
  "kerning",
  "keypoints",
  "keysplines",
  "keytimes",
  "lang",
  "lengthadjust",
  "letter-spacing",
  "kernelmatrix",
  "kernelunitlength",
  "lighting-color",
  "local",
  "marker-end",
  "marker-mid",
  "marker-start",
  "markerheight",
  "markerunits",
  "markerwidth",
  "maskcontentunits",
  "maskunits",
  "max",
  "mask",
  "media",
  "method",
  "mode",
  "min",
  "name",
  "numoctaves",
  "offset",
  "operator",
  "opacity",
  "order",
  "orient",
  "orientation",
  "origin",
  "overflow",
  "paint-order",
  "path",
  "pathlength",
  "patterncontentunits",
  "patterntransform",
  "patternunits",
  "points",
  "preservealpha",
  "preserveaspectratio",
  "primitiveunits",
  "r",
  "rx",
  "ry",
  "radius",
  "refx",
  "refy",
  "repeatcount",
  "repeatdur",
  "restart",
  "result",
  "rotate",
  "scale",
  "seed",
  "shape-rendering",
  "specularconstant",
  "specularexponent",
  "spreadmethod",
  "startoffset",
  "stddeviation",
  "stitchtiles",
  "stop-color",
  "stop-opacity",
  "stroke-dasharray",
  "stroke-dashoffset",
  "stroke-linecap",
  "stroke-linejoin",
  "stroke-miterlimit",
  "stroke-opacity",
  "stroke",
  "stroke-width",
  "style",
  "surfacescale",
  "systemlanguage",
  "tabindex",
  "targetx",
  "targety",
  "transform",
  "transform-origin",
  "text-anchor",
  "text-decoration",
  "text-rendering",
  "textlength",
  "type",
  "u1",
  "u2",
  "unicode",
  "values",
  "viewbox",
  "visibility",
  "version",
  "vert-adv-y",
  "vert-origin-x",
  "vert-origin-y",
  "width",
  "word-spacing",
  "wrap",
  "writing-mode",
  "xchannelselector",
  "ychannelselector",
  "x",
  "x1",
  "x2",
  "xmlns",
  "y",
  "y1",
  "y2",
  "z",
  "zoomandpan"
], Lf = [
  "accent",
  "accentunder",
  "align",
  "bevelled",
  "close",
  "columnsalign",
  "columnlines",
  "columnspan",
  "denomalign",
  "depth",
  "dir",
  "display",
  "displaystyle",
  "encoding",
  "fence",
  "frame",
  "height",
  "href",
  "id",
  "largeop",
  "length",
  "linethickness",
  "lspace",
  "lquote",
  "mathbackground",
  "mathcolor",
  "mathsize",
  "mathvariant",
  "maxsize",
  "minsize",
  "movablelimits",
  "notation",
  "numalign",
  "open",
  "rowalign",
  "rowlines",
  "rowspacing",
  "rowspan",
  "rspace",
  "rquote",
  "scriptlevel",
  "scriptminsize",
  "scriptsizemultiplier",
  "selection",
  "separator",
  "separators",
  "stretchy",
  "subscriptshift",
  "supscriptshift",
  "symmetric",
  "voffset",
  "width",
  "xmlns"
];
If([
  Object.fromEntries(Pf.map((e) => [e, ["*"]])),
  Object.fromEntries(Cf.map((e) => [e, ["svg:*"]])),
  Object.fromEntries(Lf.map((e) => [e, ["math:*"]]))
]);
An(["touchstart", "touchmove", "touchend", "click", "keydown"]);
var Bf = /* @__PURE__ */ new Set(["$$slots", "$$events", "$$legacy"]), Rf = /* @__PURE__ */ xe("<option> </option>"), kf = /* @__PURE__ */ xe('<div class="group-picker svelte-1btgx1p"><label for="layout-active-label" class="svelte-1btgx1p">当前 Label</label> <select id="layout-active-label" class="svelte-1btgx1p"></select></div>'), Gf = /* @__PURE__ */ xe('<!> <div class="layout-editor svelte-1btgx1p"><!> <div class="toolbar svelte-1btgx1p"><button type="button" class="svelte-1btgx1p">居中归一</button> <button type="button" class="svelte-1btgx1p">居中</button> <button type="button" class="svelte-1btgx1p">适配</button> <button type="button" class="svelte-1btgx1p">找回视野</button></div> <div class="canvas-wrap svelte-1btgx1p"><canvas class="svelte-1btgx1p"></canvas></div> <div class="status svelte-1btgx1p"> </div></div>', 1);
function Df(e, t) {
  Vr(t, !0);
  const r = /* @__PURE__ */ Ei(t, Bf), n = 2048, s = 35e-5, o = [
    [0, 255, 120],
    [0, 188, 255],
    [255, 179, 0],
    [213, 94, 255],
    [255, 82, 82],
    [75, 222, 196]
  ], f = new hs(r);
  let c, d = null, h = null, m = null, T = null, w = /* @__PURE__ */ new Map(), L = /* @__PURE__ */ new Map(), R = q(!1), P = q(!1), H = q("等待版图 mask"), _ = q(Kt({ enabled: !1 })), y = q(Kt(De())), S = q(""), E = q(!1), b = q(!1), A = q("crosshair"), N = "", W = 0, Y = { x: 0, y: 0, center_x: 0, center_y: 0 }, V = { angle: 0, rotation: 0 }, Ce = null, Ge = /* @__PURE__ */ new Set();
  function De() {
    return {
      transform_version: 2,
      revision: 0,
      center_x: 0,
      center_y: 0,
      pivot_x: 0,
      pivot_y: 0,
      scale: 1,
      rotation_deg: 0,
      preview_alpha: 0.35
    };
  }
  function Me(u) {
    return JSON.parse(JSON.stringify(u || { enabled: !1 }));
  }
  function Ie(u) {
    const v = Object.assign(Object.assign({}, De()), u || {});
    return Object.assign(Object.assign({}, v), {
      center_x: Fe(v.center_x, 0),
      center_y: Fe(v.center_y, 0),
      pivot_x: Fe(v.pivot_x, 0),
      pivot_y: Fe(v.pivot_y, 0),
      scale: Ue(Fe(v.scale, 1), 0.01, 20),
      rotation_deg: it(Fe(v.rotation_deg, 0)),
      preview_alpha: Ue(Fe(v.preview_alpha, 0.35), 0, 1),
      revision: Math.max(0, Math.trunc(Fe(v.revision, 0)))
    });
  }
  function Fe(u, v) {
    const O = Number(u);
    return Number.isFinite(O) ? O : v;
  }
  function Xe(u) {
    return typeof u == "number" ? String(u) + "px" : u || "520px";
  }
  function Ue(u, v, O) {
    return Math.max(v, Math.min(O, u));
  }
  function it(u) {
    let v = ((u + 180) % 360 + 360) % 360 - 180;
    return v === -180 && (v = 180), v;
  }
  function ke(u) {
    return typeof u + ":" + String(u);
  }
  function de() {
    var u;
    return re() ? Array.isArray((u = i(_).group_view) === null || u === void 0 ? void 0 : u.groups) ? i(_).group_view.groups : [] : [];
  }
  function re() {
    return i(_).transform_mode === "label_groups" && !!i(_).group_view;
  }
  function Ve() {
    for (const u of de())
      if (ke(u.group_id) === i(S)) return u;
    return null;
  }
  function Le() {
    const u = Ve();
    return u ? String(u.label || "Label " + String(u.group_id)) : "";
  }
  function pe() {
    return Math.max(1, Number(i(_).target_width || d?.naturalWidth || 1));
  }
  function Ne() {
    return Math.max(1, Number(i(_).target_height || d?.naturalHeight || 1));
  }
  function ct() {
    var u;
    const v = (u = w.get(i(S))) === null || u === void 0 ? void 0 : u.image;
    return Math.max(1, Number(i(_).source_width || h?.naturalWidth || v?.naturalWidth || 1));
  }
  function at() {
    var u;
    const v = (u = w.get(i(S))) === null || u === void 0 ? void 0 : u.image;
    return Math.max(1, Number(i(_).source_height || h?.naturalHeight || v?.naturalHeight || 1));
  }
  function wt() {
    return Math.min(1, n / Math.max(pe(), Ne()));
  }
  function et(u = i(S)) {
    var v;
    if (re()) {
      const C = (v = w.get(u)) === null || v === void 0 ? void 0 : v.view.foreground_bbox_xyxy;
      if (Array.isArray(C) && C.length >= 4) return C.slice(0, 4).map(Number);
    }
    const O = i(_).foreground_bbox_xyxy;
    return Array.isArray(O) && O.length >= 4 ? O.slice(0, 4).map(Number) : [0, 0, ct() - 1, at() - 1];
  }
  function It(u, v, O) {
    if (!u) {
      v === W && O(null);
      return;
    }
    const C = new Image();
    C.onload = () => {
      v === W && O(C);
    }, C.onerror = () => {
      v === W && O(null);
    }, C.src = u;
  }
  function tt(u, v) {
    const O = Math.max(1, Number(i(_).source_width || u.naturalWidth || 1)), C = Math.max(1, Number(i(_).source_height || u.naturalHeight || 1)), G = document.createElement("canvas");
    G.width = O, G.height = C;
    const k = G.getContext("2d", { willReadFrequently: !0 });
    if (!k) return null;
    k.imageSmoothingEnabled = !1, k.drawImage(u, 0, 0, O, C);
    const z = k.getImageData(0, 0, O, C), K = document.createElement("canvas");
    K.width = O, K.height = C;
    const j = K.getContext("2d");
    if (!j) return null;
    const ee = j.createImageData(O, C);
    for (let oe = 0; oe < z.data.length; oe += 4) {
      const ge = Math.max(z.data[oe], z.data[oe + 1], z.data[oe + 2]);
      z.data[oe + 3] > 0 && ge >= 128 && (ee.data[oe] = v[0], ee.data[oe + 1] = v[1], ee.data[oe + 2] = v[2], ee.data[oe + 3] = 255);
    }
    return j.putImageData(ee, 0, 0), { mask: G, tint: K };
  }
  function st() {
    if (!h) {
      m = null, T = null;
      return;
    }
    const u = tt(h, o[0]);
    m = u?.mask || null, T = u?.tint || null;
  }
  function Ye(u) {
    if (!u.image) {
      u.maskCanvas = null, u.tintCanvas = null, u.ready = !1;
      return;
    }
    const v = tt(u.image, u.color);
    u.maskCanvas = v?.mask || null, u.tintCanvas = v?.tint || null, u.ready = !!v;
  }
  function Pt() {
    Ce && (clearTimeout(Ce), Ce = null);
  }
  function ae(u) {
    const v = i(_).group_view, O = Array.isArray(v?.groups) ? v.groups : [], C = i(_).group_intent, G = !!v && !!C && String(C.selection_signature || "") === String(v.selection_signature || ""), k = /* @__PURE__ */ new Map();
    if (G && Array.isArray(C?.transforms))
      for (const ge of C.transforms)
        !ge || ge.group_id === void 0 || !ge.transform || k.set(ke(ge.group_id), Ie(ge.transform));
    const z = Ie(i(_).transform), K = /* @__PURE__ */ new Map(), j = /* @__PURE__ */ new Map(), ee = /* @__PURE__ */ new Set();
    for (let ge = 0; ge < O.length; ge++) {
      const ut = O[ge];
      if (!ut || ut.group_id === void 0 || ut.group_id === null) continue;
      const Tt = ke(ut.group_id);
      if (ee.has(Tt)) continue;
      ee.add(Tt);
      const qe = {
        view: ut,
        image: null,
        maskCanvas: null,
        tintCanvas: null,
        ready: !1,
        color: o[ge % o.length]
      };
      K.set(Tt, qe), j.set(Tt, Ie(k.get(Tt) || z));
    }
    w = K, L = j, Ge = /* @__PURE__ */ new Set();
    const oe = G && C?.active_group_id !== void 0 && C.active_group_id !== null ? ke(C.active_group_id) : "";
    g(
      S,
      K.has(oe) ? oe : K.keys().next().value || "",
      !0
    ), g(y, Ie(L.get(i(S)) || z), !0), i(S) && L.set(i(S), i(y)), h = null, m = null, T = null, g(P, !1);
    for (const [ge, ut] of K)
      It(ut.view.mask_image, u, (Tt) => {
        const qe = w.get(ge);
        !qe || qe !== ut || (qe.image = Tt, Ye(qe), ve());
      });
    g(
      H,
      i(_).status || (O.length > 0 ? "已加载 " + String(O.length) + " 个 Label；当前：" + Le() : "当前选择没有可编辑 Label"),
      !0
    );
  }
  function gt(u) {
    Pt(), W += 1;
    const v = W;
    g(_, Me(u), !0), g(R, !1), w = /* @__PURE__ */ new Map(), L = /* @__PURE__ */ new Map(), g(S, ""), g(E, !1), g(b, !1), It(i(_).base_image, v, (O) => {
      d = O, g(R, !!O), ve();
    }), re() ? ae(v) : (g(y, Ie(i(_).transform), !0), g(H, i(_).status || "编辑器已加载", !0), g(P, !1), It(i(_).mask_image, v, (O) => {
      h = O, g(P, !!O), st(), ve();
    }));
  }
  Nt(() => {
    const u = JSON.stringify(f.props.value || null);
    u !== N && (N = u, gt(f.props.value));
  }), Ga(() => {
    W += 1, Pt();
  });
  function ot(u) {
    const v = Number(u.rotation_deg || 0) * Math.PI / 180, O = Number(u.scale || 1), C = Math.cos(v), G = Math.sin(v), k = O * C, z = O * G, K = Number(u.center_x || 0) - k * Number(u.pivot_x || 0) + z * Number(u.pivot_y || 0), j = Number(u.center_y || 0) - z * Number(u.pivot_x || 0) - k * Number(u.pivot_y || 0);
    return [k, z, -z, k, K, j];
  }
  function mt(u, v, O = i(y)) {
    const [C, G, k, z, K, j] = ot(O);
    return { x: C * u + k * v + K, y: G * u + z * v + j };
  }
  function fe(u, v, O = i(y)) {
    const [C, G, k, z, K, j] = ot(O), ee = C * z - G * k;
    if (Math.abs(ee) < 1e-9) return { x: -1, y: -1 };
    const oe = u - K, ge = v - j;
    return { x: (z * oe - k * ge) / ee, y: (-G * oe + C * ge) / ee };
  }
  function we(u) {
    const v = c.getBoundingClientRect();
    return {
      x: (u.clientX - v.left) / Math.max(1, v.width) * pe(),
      y: (u.clientY - v.top) / Math.max(1, v.height) * Ne()
    };
  }
  function Be(u, v, O) {
    if (!u) return !1;
    const C = Math.round(v), G = Math.round(O);
    if (C < 0 || G < 0 || C >= u.width || G >= u.height) return !1;
    const k = u.getContext("2d", { willReadFrequently: !0 });
    if (!k) return !1;
    const z = k.getImageData(C, G, 1, 1).data;
    return z[3] > 0 && Math.max(z[0], z[1], z[2]) >= 128;
  }
  function He(u) {
    return !re() || u === i(S) ? i(y) : L.get(u) || i(y);
  }
  function Ee(u) {
    g(y, Ie(u), !0), re() && i(S) && L.set(i(S), i(y));
  }
  function ht(u, v) {
    const O = fe(u, v, i(y));
    return Be(m, O.x, O.y);
  }
  function Ct(u, v, O) {
    const C = w.get(u);
    if (!C?.ready) return !1;
    const G = fe(v, O, He(u));
    return Be(C.maskCanvas, G.x, G.y);
  }
  function Pe(u, v) {
    if (i(S) && Ct(i(S), u, v)) return i(S);
    const O = de().map((C) => ke(C.group_id)).reverse();
    for (const C of O)
      if (C !== i(S) && Ct(C, u, v)) return C;
    return "";
  }
  function ce(u = i(S), v = He(u)) {
    const [O, C, G, k] = et(u);
    return [
      mt(O, C, v),
      mt(G, C, v),
      mt(G, k, v),
      mt(O, k, v)
    ];
  }
  function $t() {
    const u = ce();
    let v = u[0];
    for (const k of u)
      (k.y < v.y || Math.abs(k.y - v.y) < 1e-6 && k.x > v.x) && (v = k);
    const O = Number(i(y).rotation_deg || 0) * Math.PI / 180, C = Math.cos(O - Math.PI / 4), G = Math.sin(O - Math.PI / 4);
    return { x: v.x + C * 34, y: v.y + G * 34 };
  }
  function Qe() {
    const u = c?.getBoundingClientRect();
    return u ? Math.max(8, 14 * pe() / Math.max(1, u.width)) : 14;
  }
  function er(u, v) {
    if (re() && !i(S)) return !1;
    const O = $t(), C = Qe();
    return Math.hypot(u - O.x, v - O.y) <= C;
  }
  function ie(u) {
    const v = re() ? i(S) : "";
    Ee(Object.assign(Object.assign({}, i(y)), {
      revision: Number(i(y).revision || 0) + 1,
      origin: u,
      scale: Ue(Number(i(y).scale || 1), 0.01, 20),
      rotation_deg: it(Number(i(y).rotation_deg || 0))
    })), v && Ge.add(v);
  }
  function _t() {
    const u = [];
    for (const v of de()) {
      const O = ke(v.group_id);
      u.push({
        group_id: v.group_id,
        transform: Object.assign({}, Ie(He(O)))
      });
    }
    return u;
  }
  function Gt() {
    const u = {
      enabled: i(_).enabled,
      transform: Object.assign({}, Ie(i(y))),
      target_width: i(_).target_width,
      target_height: i(_).target_height
    };
    re() && (u.transform_mode = "label_groups", u.group_intent = i(_).group_intent ? JSON.parse(JSON.stringify(i(_).group_intent)) : null), f.props.value = u, N = JSON.stringify(u);
  }
  function se(u, v, O, C = !0) {
    var G, k;
    if (!re()) return;
    O && ie(u);
    const z = i(_).group_view, K = Ve(), j = de().filter((oe) => Ge.has(ke(oe.group_id))).map((oe) => oe.group_id), ee = Number(((G = i(_).group_intent) === null || G === void 0 ? void 0 : G.transform_set_revision) || 0);
    g(
      _,
      Object.assign(Object.assign({}, i(_)), {
        group_intent: {
          selection_signature: String(z?.selection_signature || ""),
          transform_set_revision: ee + 1,
          active_group_id: (k = K?.group_id) !== null && k !== void 0 ? k : null,
          changed_group_ids: j,
          transforms: _t()
        }
      }),
      !0
    ), g(H, v, !0), Gt(), C && (Pt(), f.dispatch("change")), ve();
  }
  function Dt(u, v, O = !0) {
    if (re()) {
      se(u, v, !0, O);
      return;
    }
    ie(u), g(
      _,
      Object.assign(Object.assign({}, i(_)), {
        enabled: !0,
        transform: Object.assign({}, i(y)),
        status: v
      }),
      !0
    ), g(H, v, !0), Gt(), O && (Pt(), f.dispatch("change")), ve();
  }
  function Q(u, v, O = 140) {
    Dt(u, v, !1), Pt(), Ce = setTimeout(
      () => {
        Ce = null, f.dispatch("change");
      },
      O
    );
  }
  function Se(u, v, O) {
    if (d && i(R)) {
      u.imageSmoothingEnabled = !0, u.drawImage(d, 0, 0, v, O);
      return;
    }
    u.fillStyle = "#f8fafc", u.fillRect(0, 0, v, O), u.fillStyle = "#64748b", u.font = "18px sans-serif", u.fillText("请先上传图像", 24, 42);
  }
  function dt(u, v, O, C) {
    var G;
    u.save(), u.globalAlpha = Ue(Number((G = C.preview_alpha) !== null && G !== void 0 ? G : 0.35), 0, 1);
    const [k, z, K, j, ee, oe] = ot(C);
    u.setTransform(v * k, v * z, v * K, v * j, v * ee, v * oe), u.imageSmoothingEnabled = !1, u.drawImage(O, 0, 0, O.width, O.height), u.restore();
  }
  function Et(u, v) {
    const O = ce();
    u.save(), u.lineJoin = "round", u.lineWidth = Math.max(2.5, v / 700), u.strokeStyle = "rgba(0,0,0,0.82)", u.beginPath(), u.moveTo(O[0].x, O[0].y);
    for (let k = 1; k < O.length; k++) u.lineTo(O[k].x, O[k].y);
    u.closePath(), u.stroke(), u.lineWidth = Math.max(1.8, v / 1e3), u.strokeStyle = "#00ff66", u.stroke();
    const C = $t(), G = O.reduce((k, z) => z.y < k.y || Math.abs(z.y - k.y) < 1e-6 && z.x > k.x ? z : k, O[0]);
    u.strokeStyle = "#0f172a", u.lineWidth = Math.max(2, v / 900), u.beginPath(), u.moveTo(G.x, G.y), u.lineTo(C.x, C.y), u.stroke(), u.fillStyle = i(b) ? "#ffb000" : "#ffffff", u.strokeStyle = "#00ff66", u.lineWidth = Math.max(2, v / 900), u.beginPath(), u.arc(C.x, C.y, Qe(), 0, Math.PI * 2), u.fill(), u.stroke(), u.restore();
  }
  function pt(u, v, O) {
    const C = w.get(v);
    if (!C) return;
    const G = ce(v, He(v)), k = Math.min(...G.map((ge) => ge.x)), z = Math.min(...G.map((ge) => ge.y)), K = String(C.view.label || "Label " + String(C.view.group_id));
    u.save(), u.font = "600 13px sans-serif";
    const j = u.measureText(K).width + 12, ee = Ue(k, 2, Math.max(2, pe() - j - 2)), oe = Ue(z - 23, 2, Math.max(2, Ne() - 22));
    u.fillStyle = O ? "rgba(15,23,42,0.92)" : "rgba(51,65,85,0.76)", u.fillRect(ee, oe, j, 20), u.fillStyle = "#ffffff", u.fillText(K, ee + 6, oe + 14), u.restore();
  }
  function ve() {
    var u, v;
    if (!c) return;
    const O = pe(), C = Ne(), G = wt();
    c.width = Math.max(1, Math.round(O * G)), c.height = Math.max(1, Math.round(C * G));
    const k = c.getContext("2d");
    if (k && (k.setTransform(G, 0, 0, G, 0, 0), k.clearRect(0, 0, O, C), Se(k, O, C), i(_).enabled !== !1)) {
      if (re()) {
        const K = de().map((j) => ke(j.group_id)).filter((j) => j !== i(S));
        i(S) && K.push(i(S));
        for (const j of K) {
          const ee = w.get(j);
          ee?.ready && ee.tintCanvas && dt(k, G, ee.tintCanvas, He(j));
        }
        k.setTransform(G, 0, 0, G, 0, 0);
        for (const j of K)
          !((u = w.get(j)) === null || u === void 0) && u.ready && pt(k, j, j === i(S));
        i(S) && (!((v = w.get(i(S))) === null || v === void 0) && v.ready) && Et(k, O);
        return;
      }
      T && i(P) && (dt(k, G, T, i(y)), k.setTransform(G, 0, 0, G, 0, 0), Et(k, O));
    }
  }
  function Te() {
    var u;
    return i(_).enabled ? re() ? !!i(S) && !!(!((u = w.get(i(S))) === null || u === void 0) && u.ready) : i(P) && !!m : !1;
  }
  function qt(u) {
    return !re() || !w.has(u) || u === i(S) ? !1 : (i(S) && L.set(i(S), Ie(i(y))), g(S, u, !0), g(y, Ie(L.get(u)), !0), L.set(u, i(y)), g(H, "当前 Label：" + Le()), ve(), !0);
  }
  function Ft(u) {
    const v = u.currentTarget.value;
    qt(v) && se("select_group", "已选择 Label：" + Le(), !1, !0);
  }
  function tr(u) {
    if (u.button !== 0) return;
    if (!Te()) {
      g(H, "请先启用并加载版图 mask"), ve();
      return;
    }
    Pt();
    const v = we(u);
    if (er(v.x, v.y)) {
      g(A, "grabbing"), g(b, !0), V = {
        angle: Math.atan2(v.y - i(y).center_y, v.x - i(y).center_x) * 180 / Math.PI,
        rotation: Number(i(y).rotation_deg || 0)
      }, c.setPointerCapture(u.pointerId);
      return;
    }
    if (re()) {
      const O = Pe(v.x, v.y);
      if (!O) {
        g(H, "请点中任一 Label mask 前景后拖动"), ve();
        return;
      }
      qt(O);
    } else if (!ht(v.x, v.y)) {
      g(H, "请点中版图 mask 前景后拖动"), ve();
      return;
    }
    g(A, "grabbing"), g(E, !0), Y = {
      x: v.x,
      y: v.y,
      center_x: Number(i(y).center_x || 0),
      center_y: Number(i(y).center_y || 0)
    }, c.setPointerCapture(u.pointerId);
  }
  function Yt(u) {
    if (!Te()) {
      g(A, "not-allowed");
      return;
    }
    const v = re() ? !!Pe(u.x, u.y) : ht(u.x, u.y);
    if (er(u.x, u.y) || v) {
      g(A, "grab");
      return;
    }
    g(A, "crosshair");
  }
  function rr(u) {
    const v = we(u);
    if (!i(E) && !i(b)) {
      Yt(v);
      return;
    }
    if (g(A, "grabbing"), i(E))
      Ee(Object.assign(Object.assign({}, i(y)), {
        center_x: Y.center_x + v.x - Y.x,
        center_y: Y.center_y + v.y - Y.y
      })), g(H, "正在拖动 " + (re() ? Le() : "版图") + "；松开后同步变换");
    else if (i(b)) {
      const O = Math.atan2(v.y - i(y).center_y, v.x - i(y).center_x) * 180 / Math.PI;
      Ee(Object.assign(Object.assign({}, i(y)), {
        rotation_deg: it(V.rotation + O - V.angle)
      })), g(H, "正在旋转 " + (re() ? Le() : "版图") + "；松开后同步变换");
    }
    ve();
  }
  function jt() {
    !i(E) && !i(b) && g(A, "crosshair");
  }
  function Ut(u) {
    if (i(E) || i(b)) {
      g(E, !1), g(b, !1);
      try {
        c.releasePointerCapture(u.pointerId);
      } catch {
      }
      Yt(we(u));
      const v = re() ? Le() : "Canvas";
      Dt("canvas", v + " 变换已同步；点击更新预览或创建实例后使用后端权威 mask");
    }
  }
  function St(u) {
    if (!Te()) return;
    u.preventDefault();
    const v = we(u);
    if (re()) {
      const K = Pe(v.x, v.y);
      K && qt(K);
    }
    const O = fe(v.x, v.y, i(y)), C = Math.exp(-u.deltaY * s), G = Ue(Number(i(y).scale || 1) * C, 0.01, 20);
    let k = Object.assign(Object.assign({}, i(y)), { scale: G });
    const z = mt(O.x, O.y, k);
    k = Object.assign(Object.assign({}, k), {
      center_x: Number(k.center_x || 0) + v.x - z.x,
      center_y: Number(k.center_y || 0) + v.y - z.y
    }), Ee(k), Q("canvas", "滚轮缩放已同步: scale=" + G.toFixed(3));
  }
  function lt() {
    Te() && (Ee(Object.assign(Object.assign({}, i(y)), {
      center_x: pe() / 2,
      center_y: Ne() / 2,
      scale: 1,
      rotation_deg: 0
    })), Dt("reset", "居中归一 " + (re() ? Le() : "版图") + "：scale=1，rotation=0"));
  }
  function bt() {
    Te() && (Ee(Object.assign(Object.assign({}, i(y)), { center_x: pe() / 2, center_y: Ne() / 2 })), Dt("center", "居中 " + (re() ? Le() : "版图") + "：保留缩放和旋转"));
  }
  function nr() {
    if (!Te()) return;
    const [u, v, O, C] = et(), G = Math.max(1, O - u + 1), k = Math.max(1, C - v + 1), z = Ue(Math.min(pe() / G, Ne() / k) * 0.9, 0.01, 20);
    Ee(Object.assign(Object.assign({}, i(y)), {
      center_x: pe() / 2,
      center_y: Ne() / 2,
      scale: z
    })), Dt("fit", "适配 " + (re() ? Le() : "版图") + "：scale=" + z.toFixed(3));
  }
  function We() {
    if (!Te()) return;
    const u = ce(), v = Math.min(...u.map((j) => j.x)), O = Math.max(...u.map((j) => j.x)), C = Math.min(...u.map((j) => j.y)), G = Math.max(...u.map((j) => j.y));
    let k = 0, z = 0;
    const K = Math.max(20, pe() * 0.03);
    if (O < K ? k = K - O : v > pe() - K && (k = pe() - K - v), G < K ? z = K - G : C > Ne() - K && (z = Ne() - K - C), k === 0 && z === 0) {
      g(H, (re() ? Le() : "版图") + " 已经在视野内");
      return;
    }
    Ee(Object.assign(Object.assign({}, i(y)), {
      center_x: Number(i(y).center_x || 0) + k,
      center_y: Number(i(y).center_y || 0) + z
    })), Dt("bring_into_view", "已找回 " + (re() ? Le() : "版图") + " 到视野内");
  }
  {
    let u = Ke(() => i(E) || i(b) ? "focus" : "base");
    ds(e, {
      get visible() {
        return f.shared.visible;
      },
      variant: "solid",
      get border_mode() {
        return i(u);
      },
      padding: !1,
      get elem_id() {
        return f.shared.elem_id;
      },
      get elem_classes() {
        return f.shared.elem_classes;
      },
      allow_overflow: !1,
      get container() {
        return f.shared.container;
      },
      get scale() {
        return f.shared.scale;
      },
      get min_width() {
        return f.shared.min_width;
      },
      children: (v, O) => {
        var C = Gf(), G = ft(C);
        Hf(G, ni(
          {
            get autoscroll() {
              return f.shared.autoscroll;
            },
            get i18n() {
              return f.i18n;
            }
          },
          () => f.shared.loading_status,
          {
            on_clear_status: () => f.dispatch("clear_status", f.shared.loading_status)
          }
        ));
        var k = U(G, 2), z = J(k);
        {
          var K = (Vt) => {
            var ir = kf(), zt = U(J(ir), 2);
            Ar(zt, 21, de, Mr, (Ir, Pr) => {
              var Cr = Rf(), Zt = J(Cr), un = {};
              be(
                (Lr, br) => {
                  ye(Zt, Lr), un !== (un = br) && (Cr.value = (Cr.__value = br) ?? "");
                },
                [
                  () => i(Pr).label || "Label " + String(i(Pr).group_id),
                  () => ke(i(Pr).group_id)
                ]
              ), Z(Ir, Cr);
            });
            var _r;
            xi(zt), be(() => {
              _r !== (_r = i(S)) && (zt.value = (zt.__value = i(S)) ?? "", tn(zt, i(S)));
            }), te("change", zt, Ft), Z(Vt, ir);
          }, j = Ke(() => re());
          _e(z, (Vt) => {
            i(j) && Vt(K);
          });
        }
        var ee = U(z, 2), oe = J(ee), ge = U(oe, 2), ut = U(ge, 2), Tt = U(ut, 2), qe = U(ee, 2), Ze = J(qe);
        Nn(Ze, (Vt) => c = Vt, () => c);
        var on = U(qe, 2), ln = J(on);
        be(
          (Vt, ir, zt, _r, Ir) => {
            Ht(k, Vt), oe.disabled = ir, ge.disabled = zt, ut.disabled = _r, Tt.disabled = Ir, Ht(Ze, "cursor:" + i(A)), ye(ln, i(H));
          },
          [
            () => "min-height:" + Xe(f.props.height),
            () => !Te(),
            () => !Te(),
            () => !Te(),
            () => !Te()
          ]
        ), te("click", oe, lt), te("click", ge, bt), te("click", ut, nr), te("click", Tt, We), te("pointerdown", Ze, tr), te("pointermove", Ze, rr), te("pointerup", Ze, Ut), te("pointercancel", Ze, Ut), te("pointerleave", Ze, jt), te("wheel", Ze, St), Z(v, C);
      },
      $$slots: { default: !0 }
    });
  }
  zr();
}
var Ff = /* @__PURE__ */ new Set(["$$slots", "$$events", "$$legacy"]), jf = /* @__PURE__ */ xe("<option> </option>"), Uf = /* @__PURE__ */ xe('<span class="current-badge svelte-ipp6xp">当前</span>'), Vf = /* @__PURE__ */ xe('<div role="button" tabindex="0"><span class="group-drag-handle svelte-ipp6xp" title="拖动调整叠层顺序" aria-label="拖动调整叠层顺序">⠿</span> <span class="group-order svelte-ipp6xp"></span> <span class="group-reorder-label svelte-ipp6xp"> <!></span> <button type="button" class="visibility-button svelte-ipp6xp"> </button> <button type="button" title="只显示此层">S</button></div>'), zf = /* @__PURE__ */ xe('<div class="group-reorder svelte-ipp6xp" aria-label="调整 Label 叠层顺序"><div class="group-reorder-hint svelte-ipp6xp">拖动排序；眼睛与单层显示仅影响浏览器预览</div> <div class="group-reorder-list svelte-ipp6xp"></div></div>'), Xf = /* @__PURE__ */ xe('<div class="group-picker svelte-ipp6xp"><label for="layout-active-label" class="svelte-ipp6xp">当前 Label</label> <select id="layout-active-label" class="svelte-ipp6xp"></select> <div class="mode-switch svelte-ipp6xp" aria-label="显示模式"><button type="button">对齐模式</button> <button type="button">交叠预览</button></div> <!></div>'), Wf = /* @__PURE__ */ xe('<label class="svelte-ipp6xp"><span> </span> <input type="number" class="svelte-ipp6xp"/> <small class="svelte-ipp6xp"> </small></label>'), qf = /* @__PURE__ */ xe('<div class="numeric-toolbar svelte-ipp6xp" aria-label="当前层精细参数"></div>'), Yf = /* @__PURE__ */ xe('<button type="button" class="svelte-ipp6xp"> </button>'), Zf = /* @__PURE__ */ xe('<div class="proposal-actions svelte-ipp6xp"></div>'), Jf = /* @__PURE__ */ xe('<div><div class="proposal-copy svelte-ipp6xp"><strong class="svelte-ipp6xp"> </strong> <span> </span></div> <!></div>'), Kf = /* @__PURE__ */ xe('<div class="overlap-legend svelte-ipp6xp"><span class="legend-current svelte-ipp6xp"> </span> <span class="legend-next svelte-ipp6xp"> </span> <span class="legend-overlap svelte-ipp6xp"> </span></div>'), Qf = /* @__PURE__ */ xe('<div class="layout-editor svelte-ipp6xp"><!> <div class="toolbar svelte-ipp6xp"><button type="button" class="svelte-ipp6xp">居中归一</button> <button type="button" class="svelte-ipp6xp">居中</button> <button type="button" class="svelte-ipp6xp">适配</button> <button type="button" class="svelte-ipp6xp">找回标注</button> <button type="button" class="svelte-ipp6xp">画布 −</button> <button type="button" class="svelte-ipp6xp"> </button> <button type="button" class="svelte-ipp6xp">画布 ＋</button></div> <!> <!> <div class="canvas-wrap svelte-ipp6xp"><canvas class="svelte-ipp6xp"></canvas></div> <!> <div class="status svelte-ipp6xp"><span> </span> <span class="svelte-ipp6xp"> </span></div></div>');
function $f(e, t) {
  Vr(t, !0);
  const r = /* @__PURE__ */ Ei(t, Ff), n = 2048, s = Math.log(1.01) / 100, o = 250, f = 1e4, c = [
    [0, 255, 120],
    [0, 188, 255],
    [255, 179, 0],
    [213, 94, 255],
    [255, 82, 82],
    [75, 222, 196]
  ], d = new hs(r);
  let h, m = null, T = null, w = null, L = null, R = /* @__PURE__ */ new Map(), P = /* @__PURE__ */ new Map(), H = q(0), _ = null, y = q(!1), S = q(!1), E = q("等待版图 mask"), b = q(Kt({ enabled: !1 })), A = q(Kt(ht())), N = q(""), W = q(!1), Y = q(!1), V = q("crosshair"), Ce = "", Ge = 0, De = { x: 0, y: 0, center_x: 0, center_y: 0 }, Me = { angle: 0, rotation: 0 }, Ie = null, Fe = null, Xe = null, Ue = null, it = null, ke = 0, de = 0, re = "", Ve = "", Le = "", pe = /* @__PURE__ */ new Set(), Ne = q(""), ct = q(""), at = q("before"), wt = q(!1), et = q("align"), It = q(Kt(/* @__PURE__ */ new Set())), tt = q(Kt(/* @__PURE__ */ new Set())), st = q(""), Ye = q(null), Pt = /* @__PURE__ */ new Set(), ae = q(1), gt = q(0), ot = q(0), mt = q(!1), fe = q(!1), we = { client_x: 0, client_y: 0, pan_x: 0, pan_y: 0 }, Be = null, He = q(""), Ee = q(Kt({ x: "0", y: "0", scale: "100", rotation: "0", alpha: "22" }));
  function ht() {
    return {
      transform_version: 2,
      revision: 0,
      center_x: 0,
      center_y: 0,
      pivot_x: 0,
      pivot_y: 0,
      scale: 1,
      rotation_deg: 0,
      preview_alpha: 0.35
    };
  }
  function Ct(a) {
    return JSON.parse(JSON.stringify(a || { enabled: !1 }));
  }
  function Pe(a) {
    const l = Object.assign(Object.assign({}, ht()), a || {});
    return Object.assign(Object.assign({}, l), {
      center_x: ce(l.center_x, 0),
      center_y: ce(l.center_y, 0),
      pivot_x: ce(l.pivot_x, 0),
      pivot_y: ce(l.pivot_y, 0),
      scale: Qe(ce(l.scale, 1), 0.01, 20),
      rotation_deg: er(ce(l.rotation_deg, 0)),
      preview_alpha: Qe(ce(l.preview_alpha, 0.35), 0, 1),
      revision: Math.max(0, Math.trunc(ce(l.revision, 0)))
    });
  }
  function ce(a, l) {
    const p = Number(a);
    return Number.isFinite(p) ? p : l;
  }
  function $t(a) {
    return typeof a == "number" ? String(a) + "px" : a || "520px";
  }
  function Qe(a, l, p) {
    return Math.max(l, Math.min(p, a));
  }
  function er(a) {
    let l = ((a + 180) % 360 + 360) % 360 - 180;
    return l === -180 && (l = 180), l;
  }
  function ie(a) {
    return typeof a + ":" + String(a);
  }
  function _t() {
    var a;
    return se() ? Array.isArray((a = i(b).group_view) === null || a === void 0 ? void 0 : a.groups) ? i(b).group_view.groups : [] : [];
  }
  function Gt() {
    var a;
    return se() && ((a = i(b).group_view) === null || a === void 0 ? void 0 : a.allow_reorder) === !0;
  }
  function se() {
    return i(b).transform_mode === "label_groups" && !!i(b).group_view;
  }
  function Dt() {
    for (const a of _t())
      if (ie(a.group_id) === i(N)) return a;
    return null;
  }
  function Q() {
    const a = Dt();
    return a ? String(a.label || "Label " + String(a.group_id)) : "";
  }
  function Se() {
    const a = _t(), l = a.findIndex((p) => ie(p.group_id) === i(N));
    return l < 0 || l + 1 >= a.length ? "" : ie(a[l + 1].group_id);
  }
  function dt(a) {
    return i(st) ? a === i(st) : a === i(N) ? !0 : i(et) === "overlap" && a === Se();
  }
  function Et(a) {
    return i(It).has(a) ? !1 : i(tt).has(a) ? !0 : dt(a);
  }
  function pt() {
    return _t().map((a) => ie(a.group_id)).filter((a) => Et(a));
  }
  function ve() {
    return Math.max(1, Number(i(b).target_width || m?.naturalWidth || 1));
  }
  function Te() {
    return Math.max(1, Number(i(b).target_height || m?.naturalHeight || 1));
  }
  function qt() {
    var a;
    const l = (a = R.get(i(N))) === null || a === void 0 ? void 0 : a.image;
    return Math.max(1, Number(i(b).source_width || T?.naturalWidth || l?.naturalWidth || 1));
  }
  function Ft() {
    var a;
    const l = (a = R.get(i(N))) === null || a === void 0 ? void 0 : a.image;
    return Math.max(1, Number(i(b).source_height || T?.naturalHeight || l?.naturalHeight || 1));
  }
  function tr() {
    return Math.min(1, n / Math.max(ve(), Te()));
  }
  function Yt(a = i(N)) {
    var l;
    if (se()) {
      const x = (l = R.get(a)) === null || l === void 0 ? void 0 : l.view.foreground_bbox_xyxy;
      if (Array.isArray(x) && x.length >= 4) return x.slice(0, 4).map(Number);
    }
    const p = i(b).foreground_bbox_xyxy;
    return Array.isArray(p) && p.length >= 4 ? p.slice(0, 4).map(Number) : [0, 0, qt() - 1, Ft() - 1];
  }
  function rr(a, l, p) {
    if (!a) {
      l === Ge && p(null);
      return;
    }
    const x = new Image();
    x.onload = () => {
      l === Ge && p(x);
    }, x.onerror = () => {
      l === Ge && p(null);
    }, x.src = a;
  }
  function jt(a, l) {
    const p = Math.max(1, Number(i(b).source_width || a.naturalWidth || 1)), x = Math.max(1, Number(i(b).source_height || a.naturalHeight || 1)), M = document.createElement("canvas");
    M.width = p, M.height = x;
    const I = M.getContext("2d", { willReadFrequently: !0 });
    if (!I) return null;
    I.imageSmoothingEnabled = !1, I.drawImage(a, 0, 0, p, x);
    const B = I.getImageData(0, 0, p, x), D = document.createElement("canvas");
    D.width = p, D.height = x;
    const F = D.getContext("2d");
    if (!F) return null;
    const $ = F.createImageData(p, x);
    for (let ne = 0; ne < B.data.length; ne += 4) {
      const Oe = Math.max(B.data[ne], B.data[ne + 1], B.data[ne + 2]);
      B.data[ne + 3] > 0 && Oe >= 128 && ($.data[ne] = l[0], $.data[ne + 1] = l[1], $.data[ne + 2] = l[2], $.data[ne + 3] = 255);
    }
    return F.putImageData($, 0, 0), { mask: M, tint: D };
  }
  function Ut() {
    if (!T) {
      w = null, L = null;
      return;
    }
    const a = jt(T, c[0]);
    w = a?.mask || null, L = a?.tint || null;
  }
  function St(a) {
    if (!a.image) {
      a.maskCanvas = null, a.tintCanvas = null, a.ready = !1;
      return;
    }
    const l = jt(a.image, a.color);
    a.maskCanvas = l?.mask || null, a.tintCanvas = l?.tint || null, a.ready = !!l;
  }
  function lt() {
    Ie && (clearTimeout(Ie), Ie = null);
  }
  function bt() {
    Fe && (clearTimeout(Fe), Fe = null);
  }
  function nr(a) {
    const l = JSON.stringify(a);
    let p = 2166136261;
    for (let x = 0; x < l.length; x++)
      p ^= l.charCodeAt(x), p = Math.imul(p, 16777619);
    return "fnv1a32:" + (p >>> 0).toString(16).padStart(8, "0");
  }
  function We() {
    Be === null && (Be = requestAnimationFrame(() => {
      Be = null, Br();
    }));
  }
  function u() {
    i(He) || g(
      Ee,
      {
        x: ce(i(A).center_x, 0).toFixed(2).replace(/\.00$/, ""),
        y: ce(i(A).center_y, 0).toFixed(2).replace(/\.00$/, ""),
        scale: (ce(i(A).scale, 1) * 100).toFixed(3).replace(/0+$/, "").replace(/\.$/, ""),
        rotation: ce(i(A).rotation_deg, 0).toFixed(3).replace(/0+$/, "").replace(/\.$/, ""),
        alpha: (ce(i(A).preview_alpha, 0.22) * 100).toFixed(1).replace(/\.0$/, "")
      },
      !0
    );
  }
  function v(a, l) {
    if (!Array.isArray(l) || l.length !== a.length) return a;
    const p = /* @__PURE__ */ new Map();
    for (const I of a) {
      if (!I || I.group_id === void 0 || I.group_id === null) return a;
      const B = ie(I.group_id);
      if (p.has(B)) return a;
      p.set(B, I);
    }
    const x = [], M = /* @__PURE__ */ new Set();
    for (const I of l) {
      if (typeof I != "string" && typeof I != "number") return a;
      const B = ie(I), D = p.get(B);
      if (!D || M.has(B)) return a;
      M.add(B), x.push(D);
    }
    return x.length === a.length ? x : a;
  }
  function O(a) {
    var l;
    const p = i(b).group_view, x = Array.isArray(p?.groups) ? p.groups : [], M = i(b).group_intent, I = !!p && !!M && String(M.selection_signature || "") === String(p.selection_signature || "");
    re = String(i(b).editor_session_id || M?.editor_session_id || ""), Ve = String(i(b).document_epoch || M?.document_epoch || p?.selection_signature || ""), ke = Math.max(0, Math.trunc(ce(
      (l = M?.client_revision) !== null && l !== void 0 ? l : M?.transform_set_revision,
      0
    ))), de = ke, Xe = null, Ue = null, bt();
    const B = p?.allow_reorder === !0 && I ? v(x, M?.ordered_group_ids) : x;
    B !== x && p && g(
      b,
      Object.assign(Object.assign({}, i(b)), {
        group_view: Object.assign(Object.assign({}, p), { groups: B })
      }),
      !0
    );
    const D = /* @__PURE__ */ new Map();
    if (I && Array.isArray(M?.transforms))
      for (const Ae of M.transforms)
        !Ae || Ae.group_id === void 0 || !Ae.transform || D.set(ie(Ae.group_id), Pe(Ae.transform));
    const F = Pe(i(b).transform), $ = /* @__PURE__ */ new Map(), ne = /* @__PURE__ */ new Map(), Oe = /* @__PURE__ */ new Set();
    for (let Ae = 0; Ae < B.length; Ae++) {
      const Lt = B[Ae];
      if (!Lt || Lt.group_id === void 0 || Lt.group_id === null) continue;
      const ar = ie(Lt.group_id);
      if (Oe.has(ar)) continue;
      Oe.add(ar);
      const sr = {
        view: Lt,
        image: null,
        maskCanvas: null,
        tintCanvas: null,
        ready: !1,
        color: c[Ae % c.length]
      };
      $.set(ar, sr), ne.set(ar, Pe(D.get(ar) || F));
    }
    R = $, P = ne, pe = /* @__PURE__ */ new Set(), Pt = /* @__PURE__ */ new Set(), g(Ye, null);
    const me = I && M?.active_group_id !== void 0 && M.active_group_id !== null ? ie(M.active_group_id) : "";
    g(
      N,
      $.has(me) ? me : $.keys().next().value || "",
      !0
    ), g(A, Pe(P.get(i(N)) || F), !0), i(N) && P.set(i(N), i(A)), u(), T = null, w = null, L = null, g(S, !1);
    for (const [Ae, Lt] of $)
      rr(Lt.view.mask_image, a, (ar) => {
        const sr = R.get(Ae);
        !sr || sr !== Lt || (sr.image = ar, St(sr), g(H, i(H) + 1), We());
      });
    g(
      E,
      i(b).status || (B.length > 0 ? "已加载 " + String(B.length) + " 个 Label；当前：" + Q() : "当前选择没有可编辑 Label"),
      !0
    );
  }
  function C(a) {
    lt(), bt(), Xe = null, Ue = null, it = null, Ge += 1;
    const l = Ge;
    g(b, Ct(a), !0), g(y, !1), R = /* @__PURE__ */ new Map(), P = /* @__PURE__ */ new Map(), g(N, ""), g(W, !1), g(Y, !1), rr(i(b).base_image, l, (p) => {
      m = p, g(y, !!p), Br();
    }), se() ? O(l) : (g(A, Pe(i(b).transform), !0), u(), g(E, i(b).status || "编辑器已加载", !0), g(S, !1), rr(i(b).mask_image, l, (p) => {
      T = p, g(S, !!p), Ut(), Br();
    }));
  }
  function G() {
    return Array.isArray(i(Ye) === null || i(Ye) === void 0 ? void 0 : i(Ye).candidates) ? i(Ye).candidates : [];
  }
  function k(a, l = !1) {
    if (!i(Ye) || i(Ye).target_layer_id === void 0) return;
    const p = ie(i(Ye).target_layer_id);
    if (!P.has(p)) return;
    if (l && (Pt.has(p) || ce(qe(p).revision, 0) !== ce(i(Ye).start_transform_revision, 0))) {
      g(E, "粗对齐结果已到达，但检测到手动修改，未覆盖当前变换"), Zt();
      return;
    }
    Yr(p);
    const x = Pe(qe(p));
    Ze(Object.assign(Object.assign(Object.assign({}, x), a.transform), { preview_alpha: x.preview_alpha }), !1), g(Ye, null), fn("boundary_alignment", (l ? "已自动应用 " : "已应用 ") + String(a.candidate_id || "候选") + "；停止 250ms 后同步");
  }
  function z(a) {
    const l = String(a.editor_session_id || ""), p = String(a.document_epoch || "");
    if (l && l !== re || p && p !== Ve) return;
    g(Ye, Ct(a), !0), g(E, String(a.status || "粗对齐候选已生成"), !0), Zt();
    const x = Array.isArray(a.candidates) ? a.candidates : [];
    a.auto_apply && x.length ? k(x[0], !0) : We();
  }
  function K(a) {
    var l;
    const p = String(a.editor_session_id || ""), x = String(a.document_epoch || "");
    if (p !== re || x !== Ve)
      return;
    if (a.payload_kind === "ack_error") {
      bt(), Xe = null, Ue = null, g(E, String(a.status || "同步失败；本地修改仍保留"), !0), Zt(), Br();
      return;
    }
    const M = Math.max(0, Math.trunc(ce(a.server_ack_revision, 0))), I = String(a.payload_hash || "");
    Xe !== null && (M !== Xe || I !== Ue) || Xe === null && (M !== ke || I !== String(((l = i(b).group_intent) === null || l === void 0 ? void 0 : l.payload_hash) || "")) || (bt(), de = Math.max(de, M), it = null, Xe = null, Ue = null, g(E, ke > de ? "本地修改待同步" : "已同步", !0), Zt(), ke > de && Lr(0), Br());
  }
  Nt(() => {
    const a = d.props.value || null, l = JSON.stringify(a);
    if (a?.payload_kind === "ack" || a?.payload_kind === "ack_error") {
      l !== Le && (Le = l, K(a));
      return;
    }
    if (a?.payload_kind === "alignment_proposal") {
      l !== Le && (Le = l, z(a));
      return;
    }
    l !== Ce && (Ce = l, C(a));
  }), Ga(() => {
    Ge += 1, lt(), bt(), Be !== null && cancelAnimationFrame(Be);
  });
  function j(a) {
    const l = Number(a.rotation_deg || 0) * Math.PI / 180, p = Number(a.scale || 1), x = Math.cos(l), M = Math.sin(l), I = p * x, B = p * M, D = Number(a.center_x || 0) - I * Number(a.pivot_x || 0) + B * Number(a.pivot_y || 0), F = Number(a.center_y || 0) - B * Number(a.pivot_x || 0) - I * Number(a.pivot_y || 0);
    return [I, B, -B, I, D, F];
  }
  function ee(a, l, p = i(A)) {
    const [x, M, I, B, D, F] = j(p);
    return { x: x * a + I * l + D, y: M * a + B * l + F };
  }
  function oe(a, l, p = i(A)) {
    const [x, M, I, B, D, F] = j(p), $ = x * B - M * I;
    if (Math.abs($) < 1e-9) return { x: -1, y: -1 };
    const ne = a - D, Oe = l - F;
    return { x: (B * ne - I * Oe) / $, y: (-M * ne + x * Oe) / $ };
  }
  function ge(a) {
    const l = h.getBoundingClientRect(), p = (a.clientX - l.left) / Math.max(1, l.width) * ve(), x = (a.clientY - l.top) / Math.max(1, l.height) * Te();
    return {
      x: (p - i(gt)) / Math.max(0.05, i(ae)),
      y: (x - i(ot)) / Math.max(0.05, i(ae))
    };
  }
  function ut(a, l) {
    a.setTransform(l * i(ae), 0, 0, l * i(ae), l * i(gt), l * i(ot));
  }
  function Tt(a, l, p) {
    if (!a) return !1;
    const x = Math.round(l), M = Math.round(p);
    if (x < 0 || M < 0 || x >= a.width || M >= a.height) return !1;
    const I = a.getContext("2d", { willReadFrequently: !0 });
    if (!I) return !1;
    const B = I.getImageData(x, M, 1, 1).data;
    return B[3] > 0 && Math.max(B[0], B[1], B[2]) >= 128;
  }
  function qe(a) {
    return !se() || a === i(N) ? i(A) : P.get(a) || i(A);
  }
  function Ze(a, l = !0) {
    g(A, Pe(a), !0), se() && i(N) && (P.set(i(N), i(A)), l && Pt.add(i(N))), u();
  }
  function on(a, l) {
    const p = oe(a, l, i(A));
    return Tt(w, p.x, p.y);
  }
  function ln(a, l, p) {
    const x = R.get(a);
    if (!x?.ready) return !1;
    const M = oe(l, p, qe(a));
    return Tt(x.maskCanvas, M.x, M.y);
  }
  function Vt(a, l) {
    const p = new Set(pt());
    if (i(N) && p.has(i(N)) && ln(i(N), a, l)) return i(N);
    const x = _t().map((M) => ie(M.group_id)).reverse();
    for (const M of x)
      if (M !== i(N) && p.has(M) && ln(M, a, l)) return M;
    return "";
  }
  function ir(a = i(N), l = qe(a)) {
    const [p, x, M, I] = Yt(a);
    return [
      ee(p, x, l),
      ee(M, x, l),
      ee(M, I, l),
      ee(p, I, l)
    ];
  }
  function zt() {
    const a = ir();
    let l = a[0];
    for (const B of a)
      (B.y < l.y || Math.abs(B.y - l.y) < 1e-6 && B.x > l.x) && (l = B);
    const p = Number(i(A).rotation_deg || 0) * Math.PI / 180, x = Math.cos(p - Math.PI / 4), M = Math.sin(p - Math.PI / 4), I = 34 / Math.max(0.05, i(ae));
    return { x: l.x + x * I, y: l.y + M * I };
  }
  function _r() {
    const a = h?.getBoundingClientRect();
    return a ? Math.max(8 / i(ae), 14 * ve() / Math.max(1, a.width) / i(ae)) : 14 / i(ae);
  }
  function Ir(a, l) {
    if (se() && !i(N)) return !1;
    const p = zt(), x = _r();
    return Math.hypot(a - p.x, l - p.y) <= x;
  }
  function Pr(a) {
    const l = se() ? i(N) : "";
    Ze(Object.assign(Object.assign({}, i(A)), {
      revision: Number(i(A).revision || 0) + 1,
      origin: a,
      scale: Qe(Number(i(A).scale || 1), 0.01, 20),
      rotation_deg: er(Number(i(A).rotation_deg || 0))
    })), l && pe.add(l);
  }
  function Cr() {
    const a = [];
    for (const l of _t()) {
      const p = ie(l.group_id);
      a.push({
        group_id: l.group_id,
        transform: Object.assign({}, Pe(qe(p)))
      });
    }
    return a;
  }
  function Zt() {
    var a;
    const l = {
      payload_kind: "intent",
      editor_session_id: re,
      document_epoch: Ve,
      payload_hash: String(((a = i(b).group_intent) === null || a === void 0 ? void 0 : a.payload_hash) || ""),
      enabled: i(b).enabled,
      transform: Object.assign({}, Pe(i(A))),
      target_width: i(b).target_width,
      target_height: i(b).target_height
    };
    se() && (l.transform_mode = "label_groups", l.group_intent = i(b).group_intent ? JSON.parse(JSON.stringify(i(b).group_intent)) : null), d.props.value = l, Ce = JSON.stringify(l);
  }
  function un() {
    if (!se()) {
      d.dispatch("change");
      return;
    }
    const a = i(b).group_intent, l = Math.max(0, Math.trunc(ce(a?.client_revision, 0)));
    !a || l <= de || Xe !== null || (bt(), Xe = l, Ue = String(a.payload_hash || ""), g(E, "同步中…"), Zt(), d.dispatch("change"), Fe = setTimeout(
      () => {
        if (Xe !== l) return;
        Xe = null, Ue = null;
        const p = it !== l;
        it = l, g(E, p ? "同步超时，正在重试；本地修改仍保留" : "同步失败；可继续编辑或直接保存重试", !0), Zt(), Br(), p && Lr(1e3);
      },
      f
    ));
  }
  function Lr(a = o) {
    lt(), Ie = setTimeout(
      () => {
        Ie = null, un();
      },
      Math.max(0, a)
    );
  }
  function br(a, l, p, x = !0) {
    var M;
    if (!se()) return;
    p && Pr(a);
    const I = i(b).group_view, B = Dt(), D = _t().filter((ne) => pe.has(ie(ne.group_id))).map((ne) => ne.group_id), F = ke + 1, $ = Object.assign(
      Object.assign(
        {
          selection_signature: String(I?.selection_signature || Ve),
          transform_set_revision: F,
          editor_session_id: re,
          document_epoch: Ve,
          client_revision: F,
          payload_hash: "",
          active_group_id: (M = B?.group_id) !== null && M !== void 0 ? M : null,
          changed_group_ids: D
        },
        Gt() ? {
          ordered_group_ids: _t().map((ne) => ne.group_id)
        } : {}
      ),
      { transforms: Cr() }
    );
    $.payload_hash = nr(Object.assign(Object.assign({}, $), { payload_hash: "" })), ke = F, it = null, g(
      b,
      Object.assign(Object.assign({}, i(b)), {
        payload_kind: "intent",
        editor_session_id: re,
        document_epoch: Ve,
        payload_hash: $.payload_hash,
        group_intent: $
      }),
      !0
    ), g(E, l + "（本地修改）"), Zt(), x && Lr(0), We();
  }
  function yr(a, l, p = !0) {
    if (se()) {
      br(a, l, !0, p);
      return;
    }
    Pr(a), g(
      b,
      Object.assign(Object.assign({}, i(b)), {
        enabled: !0,
        transform: Object.assign({}, i(A)),
        status: l
      }),
      !0
    ), g(E, l, !0), Zt(), p && (lt(), d.dispatch("change")), We();
  }
  function fn(a, l, p = o) {
    if (se()) {
      yr(a, l, !1), Lr(p);
      return;
    }
    yr(a, l, !1), lt(), Ie = setTimeout(
      () => {
        Ie = null, d.dispatch("change");
      },
      p
    );
  }
  function vs(a, l, p) {
    if (m && i(y)) {
      a.imageSmoothingEnabled = !0, a.drawImage(m, 0, 0, l, p);
      return;
    }
    a.fillStyle = "#f8fafc", a.fillRect(0, 0, l, p), a.fillStyle = "#64748b", a.font = "18px sans-serif", a.fillText("请先上传图像", 24, 42);
  }
  function cn(a, l, p, x, M) {
    var I;
    M === void 0 && (M = Qe(Number((I = x.preview_alpha) !== null && I !== void 0 ? I : 0.35), 0, 1)), a.save(), a.globalAlpha = M;
    const [B, D, F, $, ne, Oe] = j(x);
    a.setTransform(l * i(ae) * B, l * i(ae) * D, l * i(ae) * F, l * i(ae) * $, l * (i(ae) * ne + i(gt)), l * (i(ae) * Oe + i(ot))), a.imageSmoothingEnabled = !1, a.drawImage(p, 0, 0, p.width, p.height), a.restore();
  }
  function gs(a, l, p, x) {
    const M = R.get(p), I = R.get(x);
    if (!M?.tintCanvas || !I?.tintCanvas || !M.ready || !I.ready) return;
    const B = _ || document.createElement("canvas");
    _ = B, B.width !== h.width && (B.width = h.width), B.height !== h.height && (B.height = h.height);
    const D = B.getContext("2d");
    if (D) {
      D.setTransform(1, 0, 0, 1, 0, 0), D.globalCompositeOperation = "source-over", D.clearRect(0, 0, B.width, B.height), cn(D, l, M.tintCanvas, qe(p), 1), D.globalCompositeOperation = "destination-in", cn(D, l, I.tintCanvas, qe(x), 1), D.globalCompositeOperation = "source-in", D.setTransform(1, 0, 0, 1, 0, 0), D.fillStyle = "#ff3fa4", D.fillRect(0, 0, B.width, B.height), D.globalCompositeOperation = "source-atop", D.strokeStyle = "rgba(255,255,255,0.78)", D.lineWidth = 1.5;
      for (let F = -B.height; F < B.width; F += 14)
        D.beginPath(), D.moveTo(F, 0), D.lineTo(F + B.height, B.height), D.stroke();
      a.save(), a.setTransform(1, 0, 0, 1, 0, 0), a.globalAlpha = 0.62, a.drawImage(B, 0, 0), a.restore();
    }
  }
  function Mi(a, l) {
    const p = ir();
    a.save(), a.lineJoin = "round", a.lineWidth = Math.max(2.5, l / 700) / i(ae), a.strokeStyle = "rgba(0,0,0,0.82)", a.beginPath(), a.moveTo(p[0].x, p[0].y);
    for (let I = 1; I < p.length; I++) a.lineTo(p[I].x, p[I].y);
    a.closePath(), a.stroke(), a.lineWidth = Math.max(1.8, l / 1e3) / i(ae), a.strokeStyle = "#00ff66", a.stroke();
    const x = zt(), M = p.reduce((I, B) => B.y < I.y || Math.abs(B.y - I.y) < 1e-6 && B.x > I.x ? B : I, p[0]);
    a.strokeStyle = "#0f172a", a.lineWidth = Math.max(2, l / 900) / i(ae), a.beginPath(), a.moveTo(M.x, M.y), a.lineTo(x.x, x.y), a.stroke(), a.fillStyle = i(Y) ? "#ffb000" : "#ffffff", a.strokeStyle = "#00ff66", a.lineWidth = Math.max(2, l / 900) / i(ae), a.beginPath(), a.arc(x.x, x.y, _r(), 0, Math.PI * 2), a.fill(), a.stroke(), a.restore();
  }
  function ms(a, l, p) {
    const x = R.get(l);
    if (!x) return;
    const M = ir(l, qe(l)), I = Math.min(...M.map((me) => me.x)), B = Math.min(...M.map((me) => me.y)), D = String(x.view.label || "Label " + String(x.view.group_id)), F = 1 / Math.max(0.05, i(ae));
    a.save(), a.font = "600 " + String(13 * F) + "px sans-serif";
    const $ = a.measureText(D).width + 12 * F, ne = Qe(I, 2, Math.max(2, ve() - $ - 2)), Oe = Qe(B - 23 * F, 2, Math.max(2, Te() - 22 * F));
    a.fillStyle = p ? "rgba(15,23,42,0.92)" : "rgba(51,65,85,0.62)", a.fillRect(ne, Oe, $, 20 * F), a.fillStyle = "#ffffff", a.fillText(D, ne + 6 * F, Oe + 14 * F), a.restore();
  }
  function Br() {
    var a, l, p;
    if (!h) return;
    const x = ve(), M = Te(), I = tr(), B = Math.max(1, Math.round(x * I)), D = Math.max(1, Math.round(M * I));
    h.width !== B && (h.width = B), h.height !== D && (h.height = D);
    const F = h.getContext("2d");
    if (F && (F.setTransform(1, 0, 0, 1, 0, 0), F.clearRect(0, 0, h.width, h.height), ut(F, I), vs(F, x, M), i(b).enabled !== !1)) {
      if (se()) {
        const $ = pt(), ne = $.filter((me) => me !== i(N));
        $.includes(i(N)) && ne.push(i(N));
        const Oe = Se();
        for (const me of ne) {
          const Ae = R.get(me);
          if (Ae?.ready && Ae.tintCanvas) {
            const Lt = me === i(N) ? Qe(Number((a = qe(me).preview_alpha) !== null && a !== void 0 ? a : 0.22), 0, 1) : me === Oe ? 0.1 : 0.08;
            cn(F, I, Ae.tintCanvas, qe(me), Lt);
          }
        }
        i(et) === "overlap" && i(N) && Oe && $.includes(i(N)) && $.includes(Oe) && gs(F, I, i(N), Oe), ut(F, I);
        for (const me of ne)
          !((l = R.get(me)) === null || l === void 0) && l.ready && ms(F, me, me === i(N));
        i(N) && $.includes(i(N)) && (!((p = R.get(i(N))) === null || p === void 0) && p.ready) && Mi(F, x);
        return;
      }
      L && i(S) && (cn(F, I, L, i(A)), ut(F, I), Mi(F, x));
    }
  }
  function yt() {
    var a;
    return i(b).enabled ? se() ? (i(H), !!i(N) && !!(!((a = R.get(i(N))) === null || a === void 0) && a.ready)) : i(S) && !!w : !1;
  }
  function Yr(a) {
    return !se() || !R.has(a) || a === i(N) ? !1 : (i(N) && P.set(i(N), Pe(i(A))), g(N, a, !0), g(A, Pe(P.get(a)), !0), P.set(a, i(A)), u(), g(E, "当前 Label：" + Q()), We(), !0);
  }
  function Ai(a) {
    g(et, a, !0), g(E, a === "align" ? "对齐模式：默认只显示当前层" : "交叠预览：当前层突出、下一业务层淡显", !0), We();
  }
  function _s(a, l) {
    a.stopPropagation();
    const p = new Set(i(It)), x = new Set(i(tt));
    Et(l) ? (p.add(l), x.delete(l)) : (p.delete(l), x.add(l)), g(It, p, !0), g(tt, x, !0), g(E, "显示设置仅保存在浏览器，不改变保存数据"), We();
  }
  function bs(a, l) {
    a.stopPropagation(), g(st, i(st) === l ? "" : l, !0), g(E, i(st) ? "已单独显示一层" : "已退出单层显示", !0), We();
  }
  function ys() {
    g(ae, 1), g(gt, 0), g(ot, 0), g(E, "画布视图已复位（不影响保存数据）"), We();
  }
  function Oi(a) {
    const l = Qe(i(ae) * a, 0.2, 8), p = ve() / 2, x = Te() / 2, M = (p - i(gt)) / i(ae), I = (x - i(ot)) / i(ae);
    g(gt, p - M * l), g(ot, x - I * l), g(ae, l, !0), g(E, "画布视图 " + Math.round(i(ae) * 100) + "%（仅本地）"), We();
  }
  function xs(a, l) {
    return a === "x" ? Object.assign(Object.assign({}, i(A)), { center_x: l }) : a === "y" ? Object.assign(Object.assign({}, i(A)), { center_y: l }) : a === "scale" ? Object.assign(Object.assign({}, i(A)), { scale: Qe(l / 100, 0.01, 20) }) : a === "rotation" ? Object.assign(Object.assign({}, i(A)), { rotation_deg: er(l) }) : Object.assign(Object.assign({}, i(A)), { preview_alpha: Qe(l / 100, 0, 1) });
  }
  function Ln(a) {
    const l = Number(i(Ee)[a]);
    if (g(He, ""), !Number.isFinite(l)) {
      u();
      return;
    }
    const p = Pe(i(A)), x = Pe(xs(a, l)), M = Math.abs(x.center_x - p.center_x) < 1e-9 && Math.abs(x.center_y - p.center_y) < 1e-9 && Math.abs(x.scale - p.scale) < 1e-12 && Math.abs(x.rotation_deg - p.rotation_deg) < 1e-9 && Math.abs(x.preview_alpha - p.preview_alpha) < 1e-12;
    if (Ze(x), M) {
      We();
      return;
    }
    fn("numeric", Q() + " 参数已更新");
  }
  function ws(a, l) {
    g(Ee, Object.assign(Object.assign({}, i(Ee)), { [l]: a.currentTarget.value }), !0);
  }
  function Es(a, l) {
    const p = l.key === "ArrowUp" ? 1 : -1, x = a === "x" || a === "y" || a === "alpha" ? 1 : 0.1, M = a === "x" || a === "y" || a === "alpha" ? 0.1 : 0.01, I = a === "x" || a === "y" ? 10 : a === "alpha" ? 5 : 1;
    return p * (l.shiftKey ? M : l.altKey ? I : x);
  }
  function Ss(a, l) {
    if (a.key === "Escape") {
      a.preventDefault(), g(He, ""), u(), a.currentTarget.blur();
      return;
    }
    if (a.key === "Enter") {
      a.preventDefault(), Ln(l), a.currentTarget.blur();
      return;
    }
    if (a.key === "ArrowUp" || a.key === "ArrowDown") {
      a.preventDefault();
      const p = ce(i(Ee)[l], 0);
      g(Ee, Object.assign(Object.assign({}, i(Ee)), { [l]: String(p + Es(l, a)) }), !0), Ln(l);
    }
  }
  function Ts(a) {
    a.code === "Space" && !(a.target instanceof HTMLInputElement) && !(a.target instanceof HTMLTextAreaElement) && g(fe, !0);
  }
  function Ms(a) {
    a.code === "Space" && g(fe, !1);
  }
  function Ni() {
    g(Ne, ""), g(ct, ""), g(at, "before");
  }
  function As(a, l) {
    Gt() && (g(Ne, l, !0), g(ct, ""), g(at, "before"), g(wt, !1), a.dataTransfer && (a.dataTransfer.effectAllowed = "move", a.dataTransfer.setData("text/plain", l)));
  }
  function Os(a, l) {
    if (!Gt() || !i(Ne)) return;
    a.preventDefault(), a.dataTransfer && (a.dataTransfer.dropEffect = "move");
    const x = a.currentTarget.getBoundingClientRect();
    g(ct, l, !0), g(at, a.clientY < x.top + x.height / 2 ? "before" : "after", !0);
  }
  function Ns(a, l) {
    var p;
    if (!Gt()) return;
    a.preventDefault();
    const x = i(Ne) || ((p = a.dataTransfer) === null || p === void 0 ? void 0 : p.getData("text/plain")) || "", M = i(ct) === l ? i(at) : "before", I = Is(x, l, M);
    Ni(), I && (g(wt, !0), br("reorder_groups", "已调整叠层顺序；拖动调整叠层顺序", !1, !0), setTimeout(
      () => {
        g(wt, !1);
      },
      0
    ));
  }
  function Hs() {
    Ni();
  }
  function Is(a, l, p) {
    if (!Gt() || a === l) return !1;
    const x = i(b).group_view, M = _t(), I = M.findIndex((me) => ie(me.group_id) === a), B = M.findIndex((me) => ie(me.group_id) === l);
    if (!x || I < 0 || B < 0) return !1;
    const D = M.slice(), [F] = D.splice(I, 1);
    if (!F) return !1;
    const $ = D.findIndex((me) => ie(me.group_id) === l);
    if ($ < 0) return !1;
    const ne = $ + (p === "after" ? 1 : 0);
    return D.splice(ne, 0, F), D.some((me, Ae) => ie(me.group_id) !== ie(M[Ae].group_id)) ? (g(
      b,
      Object.assign(Object.assign({}, i(b)), {
        group_view: Object.assign(Object.assign({}, x), { groups: D })
      }),
      !0
    ), !0) : !1;
  }
  function Hi(a, l) {
    if (i(wt)) {
      g(wt, !1);
      return;
    }
    const p = a.target;
    p?.closest(".group-drag-handle") || Yr(l) && br("select_group", "已选择 Label：" + Q(), !1, !0);
  }
  function Ps(a) {
    const l = a.currentTarget.value;
    Yr(l) && br("select_group", "已选择 Label：" + Q(), !1, !0);
  }
  function Ii(a) {
    g(mt, !0), g(V, "grabbing"), we = {
      client_x: a.clientX,
      client_y: a.clientY,
      pan_x: i(gt),
      pan_y: i(ot)
    }, h.setPointerCapture(a.pointerId);
  }
  function Cs(a) {
    if (a.button !== 0) return;
    lt();
    const l = ge(a);
    if (i(fe)) {
      Ii(a);
      return;
    }
    if (yt() && Ir(l.x, l.y)) {
      g(V, "grabbing"), g(Y, !0), Me = {
        angle: Math.atan2(l.y - i(A).center_y, l.x - i(A).center_x) * 180 / Math.PI,
        rotation: Number(i(A).rotation_deg || 0)
      }, h.setPointerCapture(a.pointerId);
      return;
    }
    let p = !1;
    if (yt() && se()) {
      const x = Vt(l.x, l.y);
      x && (Yr(x), p = !0);
    } else yt() && (p = on(l.x, l.y));
    if (!p) {
      Ii(a), g(E, "正在平移画布视图（仅本地）");
      return;
    }
    g(V, "grabbing"), g(W, !0), De = {
      x: l.x,
      y: l.y,
      center_x: Number(i(A).center_x || 0),
      center_y: Number(i(A).center_y || 0)
    }, h.setPointerCapture(a.pointerId);
  }
  function Bn(a) {
    if (i(fe)) {
      g(V, "grab");
      return;
    }
    if (!yt()) {
      g(V, "grab");
      return;
    }
    const l = se() ? !!Vt(a.x, a.y) : on(a.x, a.y);
    g(V, (Ir(a.x, a.y) || l, "grab"), !0);
  }
  function Ls(a) {
    if (i(mt)) {
      const p = h.getBoundingClientRect();
      g(gt, we.pan_x + (a.clientX - we.client_x) * ve() / Math.max(1, p.width)), g(ot, we.pan_y + (a.clientY - we.client_y) * Te() / Math.max(1, p.height)), g(E, "正在平移画布视图（仅本地）"), We();
      return;
    }
    const l = ge(a);
    if (!i(W) && !i(Y)) {
      Bn(l);
      return;
    }
    if (g(V, "grabbing"), i(W))
      Ze(Object.assign(Object.assign({}, i(A)), {
        center_x: De.center_x + l.x - De.x,
        center_y: De.center_y + l.y - De.y
      })), g(E, "正在拖动 " + (se() ? Q() : "版图") + "（本地）");
    else if (i(Y)) {
      const p = Math.atan2(l.y - i(A).center_y, l.x - i(A).center_x) * 180 / Math.PI;
      Ze(Object.assign(Object.assign({}, i(A)), {
        rotation_deg: er(Me.rotation + p - Me.angle)
      })), g(E, "正在旋转 " + (se() ? Q() : "版图") + "（本地）");
    }
    We();
  }
  function Bs() {
    !i(W) && !i(Y) && !i(mt) && g(V, "grab");
  }
  function Pi(a) {
    if (i(mt)) {
      g(mt, !1);
      try {
        h.releasePointerCapture(a.pointerId);
      } catch {
      }
      g(E, "画布视图已调整（仅本地，不影响保存数据）"), Bn(ge(a));
      return;
    }
    if (i(W) || i(Y)) {
      g(W, !1), g(Y, !1);
      try {
        h.releasePointerCapture(a.pointerId);
      } catch {
      }
      Bn(ge(a));
      const p = (se() ? Q() : "Canvas") + " 已在本地更新，停止操作 250ms 后静默同步";
      se() ? fn("canvas", p) : yr("canvas", p);
    }
  }
  function Ci(a) {
    return a.deltaMode === WheelEvent.DOM_DELTA_LINE ? a.deltaY * 16 : a.deltaMode === WheelEvent.DOM_DELTA_PAGE ? a.deltaY * Math.max(1, h.getBoundingClientRect().height) : a.deltaY;
  }
  function Rs(a) {
    a.preventDefault();
    const l = h.getBoundingClientRect(), p = (a.clientX - l.left) / Math.max(1, l.width) * ve(), x = (a.clientY - l.top) / Math.max(1, l.height) * Te();
    if (a.ctrlKey || a.metaKey) {
      const Oe = (p - i(gt)) / i(ae), me = (x - i(ot)) / i(ae), Ae = Qe(i(ae) * Math.exp(-Ci(a) * 15e-4), 0.2, 8);
      g(gt, p - Oe * Ae), g(ot, x - me * Ae), g(ae, Ae, !0), g(E, "画布视图 " + Math.round(i(ae) * 100) + "%（仅本地）"), We();
      return;
    }
    if (!yt()) return;
    const M = ge(a);
    if (se()) {
      const Oe = Vt(M.x, M.y);
      Oe && Yr(Oe);
    }
    const I = oe(M.x, M.y, i(A)), B = a.shiftKey ? 0.1 : a.altKey ? 5 : 1, D = Math.exp(-Ci(a) * s * B), F = Qe(Number(i(A).scale || 1) * D, 0.01, 20);
    let $ = Object.assign(Object.assign({}, i(A)), { scale: F });
    const ne = ee(I.x, I.y, $);
    $ = Object.assign(Object.assign({}, $), {
      center_x: Number($.center_x || 0) + M.x - ne.x,
      center_y: Number($.center_y || 0) + M.y - ne.y
    }), Ze($), fn("canvas", "本地缩放 " + (F * 100).toFixed(2) + "%；停止 250ms 后同步"), We();
  }
  function ks() {
    yt() && (Ze(Object.assign(Object.assign({}, i(A)), {
      center_x: ve() / 2,
      center_y: Te() / 2,
      scale: 1,
      rotation_deg: 0
    })), yr("reset", "居中归一 " + (se() ? Q() : "版图") + "：scale=1，rotation=0"));
  }
  function Gs() {
    yt() && (Ze(Object.assign(Object.assign({}, i(A)), { center_x: ve() / 2, center_y: Te() / 2 })), yr("center", "居中 " + (se() ? Q() : "版图") + "：保留缩放和旋转"));
  }
  function Ds() {
    if (!yt()) return;
    const [a, l, p, x] = Yt(), M = Math.max(1, p - a + 1), I = Math.max(1, x - l + 1), B = Qe(Math.min(ve() / M, Te() / I) * 0.9, 0.01, 20);
    Ze(Object.assign(Object.assign({}, i(A)), {
      center_x: ve() / 2,
      center_y: Te() / 2,
      scale: B
    })), yr("fit", "适配 " + (se() ? Q() : "版图") + "：scale=" + B.toFixed(3));
  }
  function Fs() {
    if (!yt()) return;
    const a = ir(), l = Math.min(...a.map((F) => F.x)), p = Math.max(...a.map((F) => F.x)), x = Math.min(...a.map((F) => F.y)), M = Math.max(...a.map((F) => F.y));
    let I = 0, B = 0;
    const D = Math.max(20, ve() * 0.03);
    if (p < D ? I = D - p : l > ve() - D && (I = ve() - D - l), M < D ? B = D - M : x > Te() - D && (B = Te() - D - x), I === 0 && B === 0) {
      g(E, (se() ? Q() : "版图") + " 已经在视野内");
      return;
    }
    Ze(Object.assign(Object.assign({}, i(A)), {
      center_x: Number(i(A).center_x || 0) + I,
      center_y: Number(i(A).center_y || 0) + B
    })), yr("bring_into_view", "已找回 " + (se() ? Q() : "版图") + " 到视野内");
  }
  te("keydown", Zi, Ts), te("keyup", Zi, Ms);
  {
    let a = Ke(() => i(W) || i(Y) || i(mt) ? "focus" : "base");
    ds(e, {
      get visible() {
        return d.shared.visible;
      },
      variant: "solid",
      get border_mode() {
        return i(a);
      },
      padding: !1,
      get elem_id() {
        return d.shared.elem_id;
      },
      get elem_classes() {
        return d.shared.elem_classes;
      },
      allow_overflow: !1,
      get container() {
        return d.shared.container;
      },
      get scale() {
        return d.shared.scale;
      },
      get min_width() {
        return d.shared.min_width;
      },
      children: (l, p) => {
        var x = Qf(), M = J(x);
        {
          var I = (Je) => {
            var xt = Xf(), rt = U(J(xt), 2);
            Ar(rt, 21, _t, Mr, (cr, hr) => {
              var Xt = jf(), Er = J(Xt), ze = {};
              be(
                (Zr, Rt) => {
                  ye(Er, Zr), ze !== (ze = Rt) && (Xt.value = (Xt.__value = Rt) ?? "");
                },
                [
                  () => i(hr).label || "Label " + String(i(hr).group_id),
                  () => ie(i(hr).group_id)
                ]
              ), Z(cr, Xt);
            });
            var nt;
            xi(rt);
            var Mt = U(rt, 2), Bt = J(Mt);
            let ur;
            var At = U(Bt, 2);
            let fr;
            var xr = U(Mt, 2);
            {
              var or = (cr) => {
                var hr = zf(), Xt = U(J(hr), 2);
                Ar(Xt, 21, _t, Mr, (Er, ze, Zr) => {
                  var Rt = Vf();
                  let Gi;
                  var hn = J(Rt);
                  mr(hn, "draggable", !0);
                  var Di = U(hn, 2);
                  Di.textContent = Zr + 1;
                  var Fi = U(Di, 2), ji = J(Fi), Js = U(ji);
                  {
                    var Ks = (je) => {
                      var kn = Uf();
                      Z(je, kn);
                    }, Qs = Ke(() => ie(i(ze).group_id) === i(N));
                    _e(Js, (je) => {
                      i(Qs) && je(Ks);
                    });
                  }
                  var dn = U(Fi, 2), $s = J(dn), Ui = U(dn, 2);
                  let Vi;
                  be(
                    (je, kn, eo, to, ro) => {
                      Gi = Ot(Rt, 1, "group-reorder-item svelte-ipp6xp", null, Gi, je), ye(ji, `${kn ?? ""} `), mr(dn, "title", eo), ye($s, to), Vi = Ot(Ui, 1, "solo-button svelte-ipp6xp", null, Vi, ro);
                    },
                    [
                      () => ({
                        active: ie(i(ze).group_id) === i(N),
                        "drop-before": i(ct) === ie(i(ze).group_id) && i(at) === "before",
                        "drop-after": i(ct) === ie(i(ze).group_id) && i(at) === "after"
                      }),
                      () => i(ze).label || "Label " + String(i(ze).group_id),
                      () => Et(ie(i(ze).group_id)) ? "隐藏此层" : "显示此层",
                      () => Et(ie(i(ze).group_id)) ? "👁" : "—",
                      () => ({
                        active: i(st) === ie(i(ze).group_id)
                      })
                    ]
                  ), te("dragstart", hn, (je) => As(je, ie(i(ze).group_id))), te("dragend", hn, Hs), te("click", dn, (je) => _s(je, ie(i(ze).group_id))), te("click", Ui, (je) => bs(je, ie(i(ze).group_id))), te("click", Rt, (je) => Hi(je, ie(i(ze).group_id))), te("keydown", Rt, (je) => {
                    (je.key === "Enter" || je.key === " ") && (je.preventDefault(), Hi(je, ie(i(ze).group_id)));
                  }), te("dragover", Rt, (je) => Os(je, ie(i(ze).group_id))), te("drop", Rt, (je) => Ns(je, ie(i(ze).group_id))), Z(Er, Rt);
                }), Z(cr, hr);
              }, wr = Ke(() => Gt());
              _e(xr, (cr) => {
                i(wr) && cr(or);
              });
            }
            be(() => {
              nt !== (nt = i(N)) && (rt.value = (rt.__value = i(N)) ?? "", tn(rt, i(N))), ur = Ot(Bt, 1, "svelte-ipp6xp", null, ur, { active: i(et) === "align" }), fr = Ot(At, 1, "svelte-ipp6xp", null, fr, { active: i(et) === "overlap" });
            }), te("change", rt, Ps), te("click", Bt, () => Ai("align")), te("click", At, () => Ai("overlap")), Z(Je, xt);
          }, B = Ke(() => se());
          _e(M, (Je) => {
            i(B) && Je(I);
          });
        }
        var D = U(M, 2), F = J(D), $ = U(F, 2), ne = U($, 2), Oe = U(ne, 2), me = U(Oe, 2), Ae = U(me, 2), Lt = J(Ae), ar = U(Ae, 2), sr = U(D, 2);
        {
          var js = (Je) => {
            var xt = qf();
            Ar(
              xt,
              20,
              () => [
                ["x", "X", "px"],
                ["y", "Y", "px"],
                ["scale", "缩放", "%"],
                ["rotation", "旋转", "°"],
                ["alpha", "透明度", "%"]
              ],
              Mr,
              (rt, nt) => {
                var Mt = Wf(), Bt = J(Mt), ur = J(Bt), At = U(Bt, 2), fr = U(At, 2), xr = J(fr);
                be(() => {
                  ye(ur, nt[1]), wl(At, i(Ee)[nt[0]]), ye(xr, nt[2]);
                }), te("focus", At, () => g(He, nt[0], !0)), te("input", At, (or) => ws(or, nt[0])), te("keydown", At, (or) => Ss(or, nt[0])), te("blur", At, () => Ln(nt[0])), Z(rt, Mt);
              }
            ), Z(Je, xt);
          }, Us = Ke(() => yt());
          _e(sr, (Je) => {
            i(Us) && Je(js);
          });
        }
        var Li = U(sr, 2);
        {
          var Vs = (Je) => {
            var xt = Jf();
            let rt;
            var nt = J(xt), Mt = J(nt), Bt = J(Mt), ur = U(Mt, 2), At = J(ur), fr = U(nt, 2);
            {
              var xr = (wr) => {
                var cr = Zf();
                Ar(cr, 21, G, Mr, (hr, Xt) => {
                  var Er = Yf(), ze = J(Er);
                  be(
                    (Zr, Rt) => ye(ze, `应用 ${i(Xt).candidate_id ?? ""}
								· 支持率 ${Zr ?? ""}%
								· 分数 ${Rt ?? ""}`),
                    [
                      () => Math.round(ce(i(Xt).metrics?.central_support, 0) * 100),
                      () => ce(i(Xt).metrics?.score, 0).toFixed(3)
                    ]
                  ), te("click", Er, () => k(i(Xt))), Z(hr, Er);
                }), Z(wr, cr);
              }, or = Ke(() => G().length);
              _e(fr, (wr) => {
                i(or) && wr(xr);
              });
            }
            be(() => {
              rt = Ot(xt, 1, "alignment-proposal svelte-ipp6xp", null, rt, { ambiguous: i(Ye).ambiguous === !0 }), ye(Bt, i(Ye).ambiguous ? "存在周期歧义" : "粗对齐建议"), ye(At, i(Ye).status);
            }), Z(Je, xt);
          };
          _e(Li, (Je) => {
            i(Ye) && Je(Vs);
          });
        }
        var Bi = U(Li, 2), lr = J(Bi);
        Nn(lr, (Je) => h = Je, () => h);
        var Ri = U(Bi, 2);
        {
          var zs = (Je) => {
            var xt = Kf(), rt = J(xt), nt = J(rt), Mt = U(rt, 2), Bt = J(Mt), ur = U(Mt, 2), At = J(ur);
            be(
              (fr, xr, or, wr) => {
                ye(nt, `当前：${fr ?? ""}`), ye(Bt, `下一层：${xr ?? ""}`), ye(At, `${or ?? ""} × ${wr ?? ""}`);
              },
              [
                () => Q(),
                () => R.get(Se())?.view.label,
                () => Q(),
                () => R.get(Se())?.view.label
              ]
            ), Z(Je, xt);
          }, Xs = Ke(() => se() && i(et) === "overlap" && Se());
          _e(Ri, (Je) => {
            i(Xs) && Je(zs);
          });
        }
        var Ws = U(Ri, 2), Rn = J(Ws);
        let ki;
        var qs = J(Rn), Ys = U(Rn, 2), Zs = J(Ys);
        be(
          (Je, xt, rt, nt, Mt, Bt) => {
            Ht(x, Je), F.disabled = xt, $.disabled = rt, ne.disabled = nt, Oe.disabled = Mt, ye(Lt, `${Bt ?? ""}%`), Ht(lr, "cursor:" + i(V)), ki = Ot(Rn, 1, "svelte-ipp6xp", null, ki, { dirty: ke > de }), ye(qs, ke > de ? Xe === null ? "本地修改" : "同步中" : "已同步"), ye(Zs, i(E));
          },
          [
            () => "min-height:" + $t(d.props.height),
            () => !yt(),
            () => !yt(),
            () => !yt(),
            () => !yt(),
            () => Math.round(i(ae) * 100)
          ]
        ), te("click", F, ks), te("click", $, Gs), te("click", ne, Ds), te("click", Oe, Fs), te("click", me, () => Oi(0.8)), te("click", Ae, ys), te("click", ar, () => Oi(1.25)), te("pointerdown", lr, Cs), te("pointermove", lr, Ls), te("pointerup", lr, Pi), te("pointercancel", lr, Pi), te("pointerleave", lr, Bs), te("wheel", lr, Rs), Z(l, x);
      },
      $$slots: { default: !0 }
    });
  }
  zr();
}
var ec = /* @__PURE__ */ new Set(["$$slots", "$$events", "$$legacy"]);
function rc(e, t) {
  Vr(t, !0);
  const r = /* @__PURE__ */ Ei(t, ec);
  var n = Tr(), s = ft(n);
  {
    var o = (c) => {
      $f(c, ni(() => r));
    }, f = (c) => {
      Df(c, ni(() => r));
    };
    _e(s, (c) => {
      t.props?.editor_mode === "overlap" ? c(o) : c(f, -1);
    });
  }
  Z(e, n), zr();
}
export {
  rc as default
};

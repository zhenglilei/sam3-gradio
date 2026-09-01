import { i as rn, g as qn, o as Ci, n as $e, u as oe, s as Ri, r as Ur, m as lt, a as x, b as l, t as nn, d as Di, q as ki, c as Wn, e as ft, f as ur, h as ir, j as Ui, T as Gi, k as Fi, l as ar, p as ut, v as an, w as ct, x as Zn, y as Yn, z as Jn, A as Gt, E as fr, B as yt, C as Qn, D as Be, F as dn, G as ji, H as Kn, I as sn, J as Vi, K as pn, L as zi, M as Xi, N as qe, O as $n, P as Ar, Q as qi, R as Wi, S as Zi, U as Yi, V as ei, W as on, X as mn, Y as gn, Z as Ji, _ as Qi, $ as Ki, a0 as $i, a1 as ea, a2 as ta, a3 as ra, a4 as na, a5 as ln, a6 as ia, a7 as Qe, a8 as Ft, a9 as aa, aa as sa, ab as oa, ac as la, ad as ua, ae as fa, af as un, ag as ca, ah as ha, ai as Oe, aj as Gr, ak as Fr, al as da, am as pa, an as kt, ao as ma, ap as ga, aq as va, ar as ba, as as _a, at as ti, au as Lt, av as V, aw as ya, ax as vn, ay as xa, az as _e, aA as cr, aB as hr, aC as Z, aD as vt, aE as re, aF as Ea, aG as se, aH as be, aI as De, aJ as wa } from "./render-DoYhCszp.js";
function ri(e) {
  throw new Error("https://svelte.dev/e/lifecycle_outside_component");
}
const Ta = [];
function Sa(e, t = !1, r = !1) {
  return tr(e, /* @__PURE__ */ new Map(), "", Ta, null, r);
}
function tr(e, t, r, n, i = null, a = !1) {
  if (typeof e == "object" && e !== null) {
    var o = t.get(e);
    if (o !== void 0) return o;
    if (e instanceof Map) return (
      /** @type {Snapshot<T>} */
      new Map(e)
    );
    if (e instanceof Set) return (
      /** @type {Snapshot<T>} */
      new Set(e)
    );
    if (rn(e)) {
      var s = (
        /** @type {Snapshot<any>} */
        Array(e.length)
      );
      t.set(e, s), i !== null && t.set(i, s);
      for (var c = 0; c < e.length; c += 1) {
        var f = e[c];
        c in e && (s[c] = tr(f, t, r, n, null, a));
      }
      return s;
    }
    if (qn(e) === Ci) {
      s = {}, t.set(e, s), i !== null && t.set(i, s);
      for (var h of Object.keys(e))
        s[h] = tr(
          // @ts-expect-error
          e[h],
          t,
          r,
          n,
          null,
          a
        );
      return s;
    }
    if (e instanceof Date)
      return (
        /** @type {Snapshot<T>} */
        structuredClone(e)
      );
    if (typeof /** @type {T & { toJSON?: any } } */
    e.toJSON == "function" && !a)
      return tr(
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
function fn(e, t, r) {
  if (e == null)
    return t(void 0), r && r(void 0), $e;
  const n = oe(
    () => e.subscribe(
      t,
      // @ts-expect-error
      r
    )
  );
  return n.unsubscribe ? () => n.unsubscribe() : n;
}
const gt = [];
function Aa(e, t) {
  return {
    subscribe: jt(e, t).subscribe
  };
}
function jt(e, t = $e) {
  let r = null;
  const n = /* @__PURE__ */ new Set();
  function i(s) {
    if (Ri(e, s) && (e = s, r)) {
      const c = !gt.length;
      for (const f of n)
        f[1](), gt.push(f, e);
      if (c) {
        for (let f = 0; f < gt.length; f += 2)
          gt[f][0](gt[f + 1]);
        gt.length = 0;
      }
    }
  }
  function a(s) {
    i(s(
      /** @type {T} */
      e
    ));
  }
  function o(s, c = $e) {
    const f = [s, c];
    return n.add(f), n.size === 1 && (r = t(i, a) || $e), s(
      /** @type {T} */
      e
    ), () => {
      n.delete(f), n.size === 0 && r && (r(), r = null);
    };
  }
  return { set: i, update: a, subscribe: o };
}
function At(e, t, r) {
  const n = !Array.isArray(e), i = n ? [e] : e;
  if (!i.every(Boolean))
    throw new Error("derived() expects stores as input, got a falsy value");
  const a = t.length < 2;
  return Aa(r, (o, s) => {
    let c = !1;
    const f = [];
    let h = 0, p = $e;
    const _ = () => {
      if (h)
        return;
      p();
      const m = t(n ? f[0] : f, o, s);
      a ? o(m) : p = typeof m == "function" ? m : $e;
    }, w = i.map(
      (m, T) => fn(
        m,
        (M) => {
          f[T] = M, h &= ~(1 << T), c && _();
        },
        () => {
          h |= 1 << T;
        }
      )
    );
    return c = !0, _(), function() {
      Ur(w), p(), c = !1;
    };
  });
}
function Ha(e) {
  let t;
  return fn(e, (r) => t = r)(), t;
}
let Kt = !1, jr = /* @__PURE__ */ Symbol("unmounted");
function bn(e, t, r) {
  const n = r[t] ??= {
    store: null,
    source: lt(void 0),
    unsubscribe: $e
  };
  if (n.store !== e && !(jr in r))
    if (n.unsubscribe(), n.store = e ?? null, e == null)
      n.source.v = void 0, n.unsubscribe = $e;
    else {
      var i = !0;
      n.unsubscribe = fn(e, (a) => {
        i ? n.source.v = a : x(n.source, a);
      }), i = !1;
    }
  return e && jr in r ? Ha(e) : l(n.source);
}
function Pa() {
  const e = {};
  function t() {
    nn(() => {
      for (var r in e)
        e[r].unsubscribe();
      Di(e, jr, {
        enumerable: !1,
        value: !0
      });
    });
  }
  return [e, t];
}
function Ia(e) {
  var t = Kt;
  try {
    return Kt = !1, [e(), Kt];
  } finally {
    Kt = t;
  }
}
function Oa(e, t) {
  if (t) {
    const r = document.body;
    e.autofocus = !0, ki(() => {
      document.activeElement === r && e.focus();
    });
  }
}
const Ba = (
  // We gotta write it like this because after downleveling the pure comment may end up in the wrong location
  globalThis?.window?.trustedTypes && /* @__PURE__ */ globalThis.window.trustedTypes.createPolicy("svelte-trusted-html", {
    /** @param {string} html */
    createHTML: (e) => e
  })
);
function Ma(e) {
  return (
    /** @type {string} */
    Ba?.createHTML(e) ?? e
  );
}
function ni(e) {
  var t = Wn("template");
  return t.innerHTML = Ma(e.replaceAll("<!>", "<!---->")), t.content;
}
function Et(e, t) {
  var r = (
    /** @type {Effect} */
    ur
  );
  r.nodes === null && (r.nodes = { start: e, end: t, a: null, t: null });
}
// @__NO_SIDE_EFFECTS__
function de(e, t) {
  var r = (t & Gi) !== 0, n = (t & Fi) !== 0, i, a = !e.startsWith("<!>");
  return () => {
    i === void 0 && (i = ni(a ? e : "<!>" + e), r || (i = /** @type {TemplateNode} */
    ir(i)));
    var o = (
      /** @type {TemplateNode} */
      n || Ui ? document.importNode(i, !0) : i.cloneNode(!0)
    );
    if (r) {
      var s = (
        /** @type {TemplateNode} */
        ir(o)
      ), c = (
        /** @type {TemplateNode} */
        o.lastChild
      );
      Et(s, c);
    } else
      Et(o, o);
    return o;
  };
}
// @__NO_SIDE_EFFECTS__
function La(e, t, r = "svg") {
  var n = !e.startsWith("<!>"), i = `<${r}>${n ? e : "<!>" + e}</${r}>`, a;
  return () => {
    if (!a) {
      var o = (
        /** @type {DocumentFragment} */
        ni(i)
      ), s = (
        /** @type {Element} */
        ir(o)
      );
      a = /** @type {Element} */
      ir(s);
    }
    var c = (
      /** @type {TemplateNode} */
      a.cloneNode(!0)
    );
    return Et(c, c), c;
  };
}
// @__NO_SIDE_EFFECTS__
function ii(e, t) {
  return /* @__PURE__ */ La(e, t, "svg");
}
function Fe(e = "") {
  {
    var t = ft(e + "");
    return Et(t, t), t;
  }
}
function _t() {
  var e = document.createDocumentFragment(), t = document.createComment(""), r = ft();
  return e.append(t, r), Et(t, r), e;
}
function D(e, t) {
  e !== null && e.before(
    /** @type {Node} */
    t
  );
}
class dr {
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
        ar(n), this.#n.delete(r);
      else {
        var i = this.#e.get(r);
        i && (ar(i.effect), this.#r.set(r, i.effect), this.#e.delete(r), i.fragment.lastChild.remove(), this.anchor.before(i.fragment), n = i.effect);
      }
      for (const [a, o] of this.#t) {
        if (this.#t.delete(a), a === t)
          break;
        const s = this.#e.get(o);
        s && (ut(s.effect), this.#e.delete(o));
      }
      for (const [a, o] of this.#r) {
        if (a === r || this.#n.has(a)) continue;
        const s = () => {
          if (Array.from(this.#t.values()).includes(a)) {
            var f = document.createDocumentFragment();
            Yn(o, f), f.append(ft()), this.#e.set(a, { effect: o, fragment: f });
          } else
            ut(o);
          this.#n.delete(a), this.#r.delete(a);
        };
        this.#i || !n ? (this.#n.add(a), an(o, s, !1)) : s();
      }
    }
  };
  /**
   * @param {Batch} batch
   */
  #s = (t) => {
    this.#t.delete(t);
    const r = Array.from(this.#t.values());
    for (const [n, i] of this.#e)
      r.includes(n) || (ut(i.effect), this.#e.delete(n));
  };
  /**
   *
   * @param {any} key
   * @param {null | ((target: TemplateNode) => void)} fn
   */
  ensure(t, r) {
    var n = (
      /** @type {Batch} */
      Zn
    ), i = Jn();
    if (r && !this.#r.has(t) && !this.#e.has(t))
      if (i) {
        var a = document.createDocumentFragment(), o = ft();
        a.append(o), this.#e.set(t, {
          effect: ct(() => r(o)),
          fragment: a
        });
      } else
        this.#r.set(
          t,
          ct(() => r(this.anchor))
        );
    if (this.#t.set(n, t), i) {
      for (const [s, c] of this.#r)
        s === t ? n.unskip_effect(c) : n.skip_effect(c);
      for (const [s, c] of this.#e)
        s === t ? n.unskip_effect(c.effect) : n.skip_effect(c.effect);
      n.oncommit(this.#a), n.ondiscard(this.#s);
    } else
      this.#a(n);
  }
}
function Na(e, t, ...r) {
  var n = new dr(e);
  Gt(() => {
    const i = t() ?? null;
    n.ensure(i, i && ((a) => i(a, ...r)));
  }, fr);
}
function ai(e) {
  yt === null && ri(), Qn && yt.l !== null ? Ra(yt).m.push(e) : Be(() => {
    const t = oe(e);
    if (typeof t == "function") return (
      /** @type {() => void} */
      t
    );
  });
}
function Ca(e) {
  yt === null && ri(), ai(() => () => oe(e));
}
function Ra(e) {
  var t = (
    /** @type {ComponentContextLegacy} */
    e.l
  );
  return t.u ??= { a: [], b: [], m: [] };
}
function $(e, t, r = !1) {
  var n = new dr(e), i = r ? fr : 0;
  function a(o, s) {
    n.ensure(o, s);
  }
  Gt(() => {
    var o = !1;
    t((s, c = 0) => {
      o = !0, a(c, s);
    }), o || a(-1, null);
  }, i);
}
function _n(e, t) {
  return t;
}
function Da(e, t, r) {
  for (var n = [], i = t.length, a, o = t.length, s = 0; s < i; s++) {
    let p = t[s];
    an(
      p,
      () => {
        if (a) {
          if (a.pending.delete(p), a.done.add(p), a.pending.size === 0) {
            var _ = (
              /** @type {Set<EachOutroGroup>} */
              e.outrogroups
            );
            Vr(e, sn(a.done)), _.delete(a), _.size === 0 && (e.outrogroups = null);
          }
        } else
          o -= 1;
      },
      !1
    );
  }
  if (o === 0) {
    var c = n.length === 0 && r !== null && e.pending.size === 0;
    if (c) {
      var f = (
        /** @type {Element} */
        r
      ), h = (
        /** @type {Element} */
        f.parentNode
      );
      Wi(h), h.append(f), e.items.clear();
    }
    Vr(e, t, !c);
  } else
    a = {
      pending: new Set(t),
      done: /* @__PURE__ */ new Set()
    }, (e.outrogroups ??= /* @__PURE__ */ new Set()).add(a);
}
function Vr(e, t, r = !0) {
  var n;
  if (e.pending.size > 0) {
    n = /* @__PURE__ */ new Set();
    for (const o of e.pending.values())
      for (const s of o)
        n.add(
          /** @type {EachItem} */
          e.items.get(s).e
        );
  }
  for (var i = 0; i < t.length; i++) {
    var a = t[i];
    if (n?.has(a)) {
      a.f |= qe;
      const o = document.createDocumentFragment();
      Yn(a, o);
    } else
      ut(t[i], r);
  }
}
var yn;
function xn(e, t, r, n, i, a = null) {
  var o = e, s = /* @__PURE__ */ new Map(), c = null, f = Kn(() => {
    var v = r();
    return (
      /** @type {V[]} */
      rn(v) ? v : v == null ? [] : sn(v)
    );
  }), h, p = /* @__PURE__ */ new Map(), _ = !0;
  function w(v) {
    (M.effect.f & $n) === 0 && (M.pending.delete(v), M.fallback = c, ka(M, h, o, t, n), c !== null && (h.length === 0 ? (c.f & qe) === 0 ? ar(c) : (c.f ^= qe, Rt(c, null, o)) : an(c, () => {
      c = null;
    })));
  }
  function m(v) {
    M.pending.delete(v);
  }
  var T = Gt(() => {
    h = /** @type {V[]} */
    l(f);
    for (var v = h.length, b = /* @__PURE__ */ new Set(), S = (
      /** @type {Batch} */
      Zn
    ), y = Jn(), E = 0; E < v; E += 1) {
      var O = h[E], P = n(O, E), I = _ ? null : s.get(P);
      I ? (I.v && dn(I.v, O), I.i && dn(I.i, E), y && S.unskip_effect(I.e)) : (I = Ua(
        s,
        _ ? o : yn ??= ft(),
        O,
        P,
        E,
        i,
        t,
        r
      ), _ || (I.e.f |= qe), s.set(P, I)), b.add(P);
    }
    if (v === 0 && a && !c && (_ ? c = ct(() => a(o)) : (c = ct(() => a(yn ??= ft())), c.f |= qe)), v > b.size && ji(), !_)
      if (p.set(S, b), y) {
        for (const [U, j] of s)
          b.has(U) || S.skip_effect(j.e);
        S.oncommit(w), S.ondiscard(m);
      } else
        w(S);
    l(f);
  }), M = { effect: T, items: s, pending: p, outrogroups: null, fallback: c };
  _ = !1;
}
function Nt(e) {
  for (; e !== null && (e.f & qi) === 0; )
    e = e.next;
  return e;
}
function ka(e, t, r, n, i) {
  var a = t.length, o = e.items, s = Nt(e.effect.first), c, f = null, h = [], p = [], _, w, m, T;
  for (T = 0; T < a; T += 1) {
    if (_ = t[T], w = i(_, T), m = /** @type {EachItem} */
    o.get(w).e, e.outrogroups !== null)
      for (const I of e.outrogroups)
        I.pending.delete(m), I.done.delete(m);
    if ((m.f & Ar) !== 0 && ar(m), (m.f & qe) !== 0)
      if (m.f ^= qe, m === s)
        Rt(m, null, r);
      else {
        var M = f ? f.next : s;
        m === e.effect.last && (e.effect.last = m.prev), m.prev && (m.prev.next = m.next), m.next && (m.next.prev = m.prev), Je(e, f, m), Je(e, m, M), Rt(m, M, r), f = m, h = [], p = [], s = Nt(f.next);
        continue;
      }
    if (m !== s) {
      if (c !== void 0 && c.has(m)) {
        if (h.length < p.length) {
          var v = p[0], b;
          f = v.prev;
          var S = h[0], y = h[h.length - 1];
          for (b = 0; b < h.length; b += 1)
            Rt(h[b], v, r);
          for (b = 0; b < p.length; b += 1)
            c.delete(p[b]);
          Je(e, S.prev, y.next), Je(e, f, S), Je(e, y, v), s = v, f = y, T -= 1, h = [], p = [];
        } else
          c.delete(m), Rt(m, s, r), Je(e, m.prev, m.next), Je(e, m, f === null ? e.effect.first : f.next), Je(e, f, m), f = m;
        continue;
      }
      for (h = [], p = []; s !== null && s !== m; )
        (c ??= /* @__PURE__ */ new Set()).add(s), p.push(s), s = Nt(s.next);
      if (s === null)
        continue;
    }
    (m.f & qe) === 0 && h.push(m), f = m, s = Nt(m.next);
  }
  if (e.outrogroups !== null) {
    for (const I of e.outrogroups)
      I.pending.size === 0 && (Vr(e, sn(I.done)), e.outrogroups?.delete(I));
    e.outrogroups.size === 0 && (e.outrogroups = null);
  }
  if (s !== null || c !== void 0) {
    var E = [];
    if (c !== void 0)
      for (m of c)
        (m.f & Ar) === 0 && E.push(m);
    for (; s !== null; )
      (s.f & Ar) === 0 && s !== e.fallback && E.push(s), s = Nt(s.next);
    var O = E.length;
    if (O > 0) {
      var P = null;
      Da(e, E, P);
    }
  }
}
function Ua(e, t, r, n, i, a, o, s) {
  var c = (o & zi) !== 0 ? (o & Xi) === 0 ? lt(r, !1, !1) : pn(r) : null, f = (o & Vi) !== 0 ? pn(i) : null;
  return {
    v: c,
    i: f,
    e: ct(() => (a(t, c ?? r, f ?? i, s), () => {
      e.delete(n);
    }))
  };
}
function Rt(e, t, r) {
  if (e.nodes)
    for (var n = e.nodes.start, i = e.nodes.end, a = t && (t.f & qe) === 0 ? (
      /** @type {EffectNodes} */
      t.nodes.start
    ) : r; n !== null; ) {
      var o = (
        /** @type {TemplateNode} */
        Zi(n)
      );
      if (a.before(n), n === i)
        return;
      n = o;
    }
}
function Je(e, t, r) {
  t === null ? e.effect.first = r : t.next = r, r === null ? e.effect.last = t : r.prev = t;
}
function zr(e, t, r, n, i) {
  var a = t.$$slots?.[r], o = !1;
  a === !0 && (a = t[r === "default" ? "children" : r], o = !0), a === void 0 || a(e, o ? () => n : n);
}
function Ga(e, t, r) {
  var n = new dr(e);
  Gt(() => {
    var i = t() ?? null;
    n.ensure(i, i && ((a) => r(a, i)));
  }, fr);
}
const Fa = () => performance.now(), ke = {
  // don't access requestAnimationFrame eagerly outside method
  // this allows basic testing of user code without JSDOM
  // bunder will eval and remove ternary when the user's app is built
  tick: (
    /** @param {any} _ */
    (e) => requestAnimationFrame(e)
  ),
  now: () => Fa(),
  tasks: /* @__PURE__ */ new Set()
};
function si() {
  const e = ke.now();
  ke.tasks.forEach((t) => {
    t.c(e) || (ke.tasks.delete(t), t.f());
  }), ke.tasks.size !== 0 && ke.tick(si);
}
function ja(e) {
  let t;
  return ke.tasks.size === 0 && ke.tick(si), {
    promise: new Promise((r) => {
      ke.tasks.add(t = { c: e, f: r });
    }),
    abort() {
      ke.tasks.delete(t);
    }
  };
}
function Va(e, t, r, n, i, a) {
  var o = null, s = (
    /** @type {TemplateNode} */
    e
  ), c = new dr(s, !1);
  Gt(() => {
    const f = t() || null;
    var h = f === "svg" ? Yi : void 0;
    if (f === null) {
      c.ensure(null, null);
      return;
    }
    return c.ensure(f, (p) => {
      if (f) {
        if (o = Wn(f, h), Et(o, o), n) {
          var _ = null, w = o.appendChild(ft());
          n(o, w), _?.remove();
        }
        ur.nodes.end = o, p.before(o);
      }
    }), () => {
    };
  }, fr), nn(() => {
  });
}
function za(e, t) {
  var r = void 0, n;
  ei(() => {
    r !== (r = t()) && (n && (ut(n), n = null), r && (n = ct(() => {
      on(() => (
        /** @type {(node: Element) => void} */
        r(e)
      ));
    })));
  });
}
function oi(e) {
  var t, r, n = "";
  if (typeof e == "string" || typeof e == "number") n += e;
  else if (typeof e == "object") if (Array.isArray(e)) {
    var i = e.length;
    for (t = 0; t < i; t++) e[t] && (r = oi(e[t])) && (n && (n += " "), n += r);
  } else for (r in e) e[r] && (n && (n += " "), n += r);
  return n;
}
function Xa() {
  for (var e, t, r = 0, n = "", i = arguments.length; r < i; r++) (e = arguments[r]) && (t = oi(e)) && (n && (n += " "), n += t);
  return n;
}
function qa(e) {
  return typeof e == "object" ? Xa(e) : e ?? "";
}
const En = [...` \t
\r\f \v\uFEFF`];
function Wa(e, t, r) {
  var n = e == null ? "" : "" + e;
  if (t && (n = n ? n + " " + t : t), r) {
    for (var i of Object.keys(r))
      if (r[i])
        n = n ? n + " " + i : i;
      else if (n.length)
        for (var a = i.length, o = 0; (o = n.indexOf(i, o)) >= 0; ) {
          var s = o + a;
          (o === 0 || En.includes(n[o - 1])) && (s === n.length || En.includes(n[s])) ? n = (o === 0 ? "" : n.substring(0, o)) + n.substring(s + 1) : o = s;
        }
  }
  return n === "" ? null : n;
}
function wn(e, t = !1) {
  var r = t ? " !important;" : ";", n = "";
  for (var i of Object.keys(e)) {
    var a = e[i];
    a != null && a !== "" && (n += " " + i + ": " + a + r);
  }
  return n;
}
function Hr(e) {
  return e[0] !== "-" || e[1] !== "-" ? e.toLowerCase() : e;
}
function Za(e, t) {
  if (t) {
    var r = "", n, i;
    if (Array.isArray(t) ? (n = t[0], i = t[1]) : n = t, e) {
      e = String(e).replaceAll(/\/\*.*?\*\//g, "").trim();
      var a = !1, o = 0, s = !1, c = [];
      n && c.push(...Object.keys(n).map(Hr)), i && c.push(...Object.keys(i).map(Hr));
      var f = 0, h = -1;
      const T = e.length;
      for (var p = 0; p < T; p++) {
        var _ = e[p];
        if (s ? _ === "/" && e[p - 1] === "*" && (s = !1) : a ? a === _ && (a = !1) : _ === "/" && e[p + 1] === "*" ? s = !0 : _ === '"' || _ === "'" ? a = _ : _ === "(" ? o++ : _ === ")" && o--, !s && a === !1 && o === 0) {
          if (_ === ":" && h === -1)
            h = p;
          else if (_ === ";" || p === T - 1) {
            if (h !== -1) {
              var w = Hr(e.substring(f, h).trim());
              if (!c.includes(w)) {
                _ !== ";" && p++;
                var m = e.substring(f, p).trim();
                r += " " + m + ";";
              }
            }
            f = p + 1, h = -1;
          }
        }
      }
    }
    return n && (r += wn(n)), i && (r += wn(i, !0)), r = r.trim(), r === "" ? null : r;
  }
  return e == null ? null : String(e);
}
function et(e, t, r, n, i, a) {
  var o = (
    /** @type {any} */
    e[mn]
  );
  if (o !== r || o === void 0) {
    var s = Wa(r, n, a);
    s == null ? e.removeAttribute("class") : t ? e.className = s : e.setAttribute("class", s), e[mn] = r;
  } else if (a && i !== a)
    for (var c in a) {
      var f = !!a[c];
      (i == null || f !== !!i[c]) && e.classList.toggle(c, f);
    }
  return a;
}
function Pr(e, t = {}, r, n) {
  for (var i in r) {
    var a = r[i];
    t[i] !== a && (r[i] == null ? e.style.removeProperty(i) : e.style.setProperty(i, a, n));
  }
}
function Ue(e, t, r, n) {
  var i = (
    /** @type {any} */
    e[gn]
  );
  if (i !== t) {
    var a = Za(t, n);
    a == null ? e.removeAttribute("style") : e.style.cssText = a, e[gn] = t;
  } else n && (Array.isArray(n) ? (Pr(e, r?.[0], n[0]), Pr(e, r?.[1], n[1], "important")) : Pr(e, r, n));
  return n;
}
function Xr(e, t, r = !1) {
  if (e.multiple) {
    if (t == null)
      return;
    if (!rn(t))
      return Ji();
    for (var n of e.options)
      n.selected = t.includes(Tn(n));
    return;
  }
  for (n of e.options) {
    var i = Tn(n);
    if (Qi(i, t)) {
      n.selected = !0;
      return;
    }
  }
  (!r || t !== void 0) && (e.selectedIndex = -1);
}
function Ya(e) {
  var t = new MutationObserver(() => {
    "__value" in e && Xr(e, e.__value);
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
  }), nn(() => {
    t.disconnect();
  });
}
function Tn(e) {
  return "__value" in e ? e.__value : e.value;
}
const Dt = /* @__PURE__ */ Symbol("class"), bt = /* @__PURE__ */ Symbol("style"), li = /* @__PURE__ */ Symbol("is custom element"), ui = /* @__PURE__ */ Symbol("is html"), Ja = ln ? "input" : "INPUT", Qa = ln ? "option" : "OPTION", Ka = ln ? "select" : "SELECT";
function $a(e, t) {
  t ? e.hasAttribute("selected") || e.setAttribute("selected", "") : e.removeAttribute("selected");
}
function xt(e, t, r, n) {
  var i = fi(e);
  i[t] !== (i[t] = r) && (t === "loading" && (e[Ki] = r), r == null ? e.removeAttribute(t) : typeof r != "string" && ci(e).includes(t) ? e[t] = r : e.setAttribute(t, r));
}
function es(e, t, r, n, i = !1, a = !1) {
  var o = fi(e), s = o[li], c = !o[ui], f = t || {}, h = e.nodeName === Qa;
  for (var p in t)
    !(p in r) && p[0] + p[1] !== "$$" && (r[p] = null);
  r.class ? r.class = qa(r.class) : r.class = null, r[bt] && (r.style ??= null);
  var _ = ci(e);
  if (e.nodeName === Ja && "type" in r && ("value" in r || "__value" in r)) {
    var w = r.type;
    (w !== f.type || w === void 0 && e.hasAttribute("type")) && (f.type = w, xt(e, "type", w));
  }
  for (const y in r) {
    let E = r[y];
    if (h && y === "value" && E == null) {
      e.value = e.__value = "", f[y] = E;
      continue;
    }
    if (y === "class") {
      var m = e.namespaceURI === "http://www.w3.org/1999/xhtml";
      et(e, m, E, n, t?.[Dt], r[Dt]), f[y] = E, f[Dt] = r[Dt];
      continue;
    }
    if (y === "style") {
      Ue(e, E, t?.[bt], r[bt]), f[y] = E, f[bt] = r[bt];
      continue;
    }
    var T = f[y];
    if (!(E === T && !(E === void 0 && e.hasAttribute(y)))) {
      f[y] = E;
      var M = y[0] + y[1];
      if (M !== "$$")
        if (M === "on") {
          const O = {}, P = "$$" + y;
          let I = y.slice(2);
          var v = la(I);
          if (ia(I) && (I = I.slice(0, -7), O.capture = !0), !v && T) {
            if (E != null) continue;
            e.removeEventListener(I, f[P], O), f[P] = null;
          }
          if (v)
            Qe(I, e, E), Ft([I]);
          else if (E != null) {
            let U = function(j) {
              f[y].call(this, j);
            };
            f[P] = aa(I, e, U, O);
          }
        } else if (y === "style")
          xt(e, y, E);
        else if (y === "autofocus")
          Oa(
            /** @type {HTMLElement} */
            e,
            !!E
          );
        else if (!s && (y === "__value" || y === "value" && E != null))
          e.value = e.__value = E;
        else if (y === "selected" && h)
          $a(
            /** @type {HTMLOptionElement} */
            e,
            E
          );
        else {
          var b = y;
          c || (b = sa(b));
          var S = b === "defaultValue" || b === "defaultChecked";
          if (E == null && !s && !S)
            if (o[y] = null, b === "value" || b === "checked") {
              let O = (
                /** @type {HTMLInputElement} */
                e
              );
              const P = t === void 0;
              if (b === "value") {
                let I = O.defaultValue;
                O.removeAttribute(b), O.defaultValue = I, O.value = O.__value = P ? I : null;
              } else {
                let I = O.defaultChecked;
                O.removeAttribute(b), O.defaultChecked = I, O.checked = P ? I : !1;
              }
            } else
              e.removeAttribute(y);
          else S || _.includes(b) && (s || typeof E != "string") ? (e[b] = E, b in o && (o[b] = oa)) : typeof E != "function" && xt(e, b, E);
        }
    }
  }
  return f;
}
function ts(e, t, r = [], n = [], i = [], a, o = !1, s = !1) {
  ra(i, r, n, (c) => {
    var f = void 0, h = {}, p = e.nodeName === Ka, _ = !1;
    if (ei(() => {
      var m = t(...c.map(l)), T = es(
        e,
        f,
        m,
        a,
        o,
        s
      );
      _ && p && "value" in m && Xr(
        /** @type {HTMLSelectElement} */
        e,
        m.value
      );
      for (let v of Object.getOwnPropertySymbols(h))
        m[v] || ut(h[v]);
      for (let v of Object.getOwnPropertySymbols(m)) {
        var M = m[v];
        v.description === na && (!f || M !== f[v]) && (h[v] && ut(h[v]), h[v] = ct(() => za(e, () => M))), T[v] = M;
      }
      f = T;
    }), p) {
      var w = (
        /** @type {HTMLSelectElement} */
        e
      );
      on(() => {
        Xr(
          w,
          /** @type {Record<string | symbol, any>} */
          f.value,
          !0
        ), Ya(w);
      });
    }
    _ = !0;
  });
}
function fi(e) {
  return (
    /** @type {Record<string | symbol, unknown>} **/
    /** @type {any} */
    e[$i] ??= {
      [li]: e.nodeName.includes("-"),
      [ui]: e.namespaceURI === ea
    }
  );
}
var Sn = /* @__PURE__ */ new Map();
function ci(e) {
  var t = e.getAttribute("is") || e.nodeName, r = Sn.get(t);
  if (r) return r;
  Sn.set(t, r = []);
  for (var n, i = e, a = Element.prototype; a !== i; ) {
    n = ta(i);
    for (var o in n)
      n[o].set && // better safe than sorry, we don't want spread attributes to mess with HTML content
      o !== "innerHTML" && o !== "textContent" && o !== "innerText" && r.push(o);
    i = qn(i);
  }
  return r;
}
function Ir(e, t) {
  return e === t || e?.[un] === t;
}
function sr(e = {}, t, r, n) {
  var i = (
    /** @type {ComponentContext} */
    yt.r
  ), a = (
    /** @type {Effect} */
    ur
  );
  return on(() => {
    var o, s;
    return ua(() => {
      o = s, s = [], oe(() => {
        Ir(r(...s), e) || (t(e, ...s), o && Ir(r(...o), e) && t(null, ...o));
      });
    }), () => {
      let c = a;
      for (; c !== i && c.parent !== null && c.parent.f & fa; )
        c = c.parent;
      const f = () => {
        s && Ir(r(...s), e) && t(null, ...s);
      }, h = c.teardown;
      c.teardown = () => {
        f(), h?.();
      };
    };
  }), e;
}
function rs(e = !1) {
  const t = (
    /** @type {ComponentContextLegacy} */
    yt
  ), r = t.l.u;
  if (!r) return;
  let n = () => Oe(t.s);
  if (e) {
    let i = 0, a = (
      /** @type {Record<string, any>} */
      {}
    );
    const o = Gr(() => {
      let s = !1;
      const c = t.s;
      for (const f in c)
        c[f] !== a[f] && (a[f] = c[f], s = !0);
      return s && i++, i;
    });
    n = () => l(o);
  }
  r.b.length && ca(() => {
    An(t, n), Ur(r.b);
  }), Be(() => {
    const i = oe(() => r.m.map(ha));
    return () => {
      for (const a of i)
        typeof a == "function" && a();
    };
  }), r.a.length && Be(() => {
    An(t, n), Ur(r.a);
  });
}
function An(e, t) {
  if (e.l.s)
    for (const r of e.l.s) l(r);
  t();
}
const ns = {
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
function is(e, t, r) {
  return new Proxy({ props: e, exclude: t }, ns);
}
const as = {
  get(e, t) {
    let r = e.props.length;
    for (; r--; ) {
      let n = e.props[r];
      if (Lt(n) && (n = n()), typeof n == "object" && n !== null && t in n) return n[t];
    }
  },
  set(e, t, r) {
    let n = e.props.length;
    for (; n--; ) {
      let i = e.props[n];
      Lt(i) && (i = i());
      const a = Fr(i, t);
      if (a && a.set)
        return a.set(r), !0;
    }
    return !1;
  },
  getOwnPropertyDescriptor(e, t) {
    let r = e.props.length;
    for (; r--; ) {
      let n = e.props[r];
      if (Lt(n) && (n = n()), typeof n == "object" && n !== null && t in n) {
        const i = Fr(n, t);
        return i && !i.configurable && (i.configurable = !0), i;
      }
    }
  },
  has(e, t) {
    if (t === un || t === ti) return !1;
    for (let r of e.props)
      if (Lt(r) && (r = r()), r != null && t in r) return !0;
    return !1;
  },
  ownKeys(e) {
    const t = [];
    for (let r of e.props)
      if (Lt(r) && (r = r()), !!r) {
        for (const n in r)
          t.includes(n) || t.push(n);
        for (const n of Object.getOwnPropertySymbols(r))
          t.includes(n) || t.push(n);
      }
    return t;
  }
};
function ss(...e) {
  return new Proxy({ props: e }, as);
}
function B(e, t, r, n) {
  var i = !Qn || (r & ga) !== 0, a = (r & ma) !== 0, o = (r & ba) !== 0, s = (
    /** @type {V} */
    n
  ), c = !0, f = (
    /** @type {Derived<V> | undefined} */
    void 0
  ), h = () => o && i ? (f ??= Gr(
    /** @type {() => V} */
    n
  ), l(f)) : (c && (c = !1, s = o ? oe(
    /** @type {() => V} */
    n
  ) : (
    /** @type {V} */
    n
  )), s);
  let p;
  if (a) {
    var _ = un in e || ti in e;
    p = Fr(e, t)?.set ?? (_ && t in e ? (y) => e[t] = y : void 0);
  }
  var w, m = !1;
  a ? [w, m] = Ia(() => (
    /** @type {V} */
    e[t]
  )) : w = /** @type {V} */
  e[t], w === void 0 && n !== void 0 && (w = h(), p && (i && da(), p(w)));
  var T;
  if (i ? T = () => {
    var y = (
      /** @type {V} */
      e[t]
    );
    return y === void 0 ? h() : (c = !0, y);
  } : T = () => {
    var y = (
      /** @type {V} */
      e[t]
    );
    return y !== void 0 && (s = /** @type {V} */
    void 0), y === void 0 ? s : y;
  }, i && (r & pa) === 0)
    return T;
  if (p) {
    var M = e.$$legacy;
    return (
      /** @type {() => V} */
      (function(y, E) {
        return arguments.length > 0 ? ((!i || !E || M || m) && p(E ? T() : y), y) : T();
      })
    );
  }
  var v = !1, b = ((r & va) !== 0 ? Gr : Kn)(() => (v = !1, T()));
  a && l(b);
  var S = (
    /** @type {Effect} */
    ur
  );
  return (
    /** @type {() => V} */
    (function(y, E) {
      if (arguments.length > 0) {
        const O = E ? l(b) : i && a ? kt(y) : y;
        return x(b, O), v = !0, s !== void 0 && (s = O), y;
      }
      return _a && v || (S.f & $n) !== 0 ? b.v : l(b);
    })
  );
}
const os = [
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
], Hn = {
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
os.reduce((e, { color: t, primary: r, secondary: n }) => ({
  ...e,
  [t]: {
    primary: Hn[t][r],
    secondary: Hn[t][n]
  }
}), {});
function ls(e) {
  return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, "default") ? e.default : e;
}
var Or, Pn;
function us() {
  if (Pn) return Or;
  Pn = 1;
  var e = function(b) {
    return t(b) && !r(b);
  };
  function t(v) {
    return !!v && typeof v == "object";
  }
  function r(v) {
    var b = Object.prototype.toString.call(v);
    return b === "[object RegExp]" || b === "[object Date]" || a(v);
  }
  var n = typeof Symbol == "function" && Symbol.for, i = n ? /* @__PURE__ */ Symbol.for("react.element") : 60103;
  function a(v) {
    return v.$$typeof === i;
  }
  function o(v) {
    return Array.isArray(v) ? [] : {};
  }
  function s(v, b) {
    return b.clone !== !1 && b.isMergeableObject(v) ? T(o(v), v, b) : v;
  }
  function c(v, b, S) {
    return v.concat(b).map(function(y) {
      return s(y, S);
    });
  }
  function f(v, b) {
    if (!b.customMerge)
      return T;
    var S = b.customMerge(v);
    return typeof S == "function" ? S : T;
  }
  function h(v) {
    return Object.getOwnPropertySymbols ? Object.getOwnPropertySymbols(v).filter(function(b) {
      return Object.propertyIsEnumerable.call(v, b);
    }) : [];
  }
  function p(v) {
    return Object.keys(v).concat(h(v));
  }
  function _(v, b) {
    try {
      return b in v;
    } catch {
      return !1;
    }
  }
  function w(v, b) {
    return _(v, b) && !(Object.hasOwnProperty.call(v, b) && Object.propertyIsEnumerable.call(v, b));
  }
  function m(v, b, S) {
    var y = {};
    return S.isMergeableObject(v) && p(v).forEach(function(E) {
      y[E] = s(v[E], S);
    }), p(b).forEach(function(E) {
      w(v, E) || (_(v, E) && S.isMergeableObject(b[E]) ? y[E] = f(E, S)(v[E], b[E], S) : y[E] = s(b[E], S));
    }), y;
  }
  function T(v, b, S) {
    S = S || {}, S.arrayMerge = S.arrayMerge || c, S.isMergeableObject = S.isMergeableObject || e, S.cloneUnlessOtherwiseSpecified = s;
    var y = Array.isArray(b), E = Array.isArray(v), O = y === E;
    return O ? y ? S.arrayMerge(v, b, S) : m(v, b, S) : s(b, S);
  }
  T.all = function(b, S) {
    if (!Array.isArray(b))
      throw new Error("first argument should be an array");
    return b.reduce(function(y, E) {
      return T(y, E, S);
    }, {});
  };
  var M = T;
  return Or = M, Or;
}
var fs = us();
const cs = /* @__PURE__ */ ls(fs);
var qr = function(e, t) {
  return qr = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(r, n) {
    r.__proto__ = n;
  } || function(r, n) {
    for (var i in n) Object.prototype.hasOwnProperty.call(n, i) && (r[i] = n[i]);
  }, qr(e, t);
};
function pr(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Class extends value " + String(t) + " is not a constructor or null");
  qr(e, t);
  function r() {
    this.constructor = e;
  }
  e.prototype = t === null ? Object.create(t) : (r.prototype = t.prototype, new r());
}
var F = function() {
  return F = Object.assign || function(t) {
    for (var r, n = 1, i = arguments.length; n < i; n++) {
      r = arguments[n];
      for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (t[a] = r[a]);
    }
    return t;
  }, F.apply(this, arguments);
};
function hs(e, t) {
  var r = {};
  for (var n in e) Object.prototype.hasOwnProperty.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && typeof Object.getOwnPropertySymbols == "function")
    for (var i = 0, n = Object.getOwnPropertySymbols(e); i < n.length; i++)
      t.indexOf(n[i]) < 0 && Object.prototype.propertyIsEnumerable.call(e, n[i]) && (r[n[i]] = e[n[i]]);
  return r;
}
function Br(e, t, r) {
  if (r || arguments.length === 2) for (var n = 0, i = t.length, a; n < i; n++)
    (a || !(n in t)) && (a || (a = Array.prototype.slice.call(t, 0, n)), a[n] = t[n]);
  return e.concat(a || Array.prototype.slice.call(t));
}
function Mr(e, t) {
  var r = t && t.cache ? t.cache : _s, n = t && t.serializer ? t.serializer : vs, i = t && t.strategy ? t.strategy : ms;
  return i(e, {
    cache: r,
    serializer: n
  });
}
function ds(e) {
  return e == null || typeof e == "number" || typeof e == "boolean";
}
function ps(e, t, r, n) {
  var i = ds(n) ? n : r(n), a = t.get(i);
  return typeof a > "u" && (a = e.call(this, n), t.set(i, a)), a;
}
function hi(e, t, r) {
  var n = Array.prototype.slice.call(arguments, 3), i = r(n), a = t.get(i);
  return typeof a > "u" && (a = e.apply(this, n), t.set(i, a)), a;
}
function di(e, t, r, n, i) {
  return r.bind(t, e, n, i);
}
function ms(e, t) {
  var r = e.length === 1 ? ps : hi;
  return di(e, this, r, t.cache.create(), t.serializer);
}
function gs(e, t) {
  return di(e, this, hi, t.cache.create(), t.serializer);
}
var vs = function() {
  return JSON.stringify(arguments);
}, bs = (
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
), _s = {
  create: function() {
    return new bs();
  }
}, Lr = {
  variadic: gs
}, R;
(function(e) {
  e[e.EXPECT_ARGUMENT_CLOSING_BRACE = 1] = "EXPECT_ARGUMENT_CLOSING_BRACE", e[e.EMPTY_ARGUMENT = 2] = "EMPTY_ARGUMENT", e[e.MALFORMED_ARGUMENT = 3] = "MALFORMED_ARGUMENT", e[e.EXPECT_ARGUMENT_TYPE = 4] = "EXPECT_ARGUMENT_TYPE", e[e.INVALID_ARGUMENT_TYPE = 5] = "INVALID_ARGUMENT_TYPE", e[e.EXPECT_ARGUMENT_STYLE = 6] = "EXPECT_ARGUMENT_STYLE", e[e.INVALID_NUMBER_SKELETON = 7] = "INVALID_NUMBER_SKELETON", e[e.INVALID_DATE_TIME_SKELETON = 8] = "INVALID_DATE_TIME_SKELETON", e[e.EXPECT_NUMBER_SKELETON = 9] = "EXPECT_NUMBER_SKELETON", e[e.EXPECT_DATE_TIME_SKELETON = 10] = "EXPECT_DATE_TIME_SKELETON", e[e.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE = 11] = "UNCLOSED_QUOTE_IN_ARGUMENT_STYLE", e[e.EXPECT_SELECT_ARGUMENT_OPTIONS = 12] = "EXPECT_SELECT_ARGUMENT_OPTIONS", e[e.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE = 13] = "EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE = 14] = "INVALID_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR = 15] = "EXPECT_SELECT_ARGUMENT_SELECTOR", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR = 16] = "EXPECT_PLURAL_ARGUMENT_SELECTOR", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT = 17] = "EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT = 18] = "EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT", e[e.INVALID_PLURAL_ARGUMENT_SELECTOR = 19] = "INVALID_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_PLURAL_ARGUMENT_SELECTOR = 20] = "DUPLICATE_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_SELECT_ARGUMENT_SELECTOR = 21] = "DUPLICATE_SELECT_ARGUMENT_SELECTOR", e[e.MISSING_OTHER_CLAUSE = 22] = "MISSING_OTHER_CLAUSE", e[e.INVALID_TAG = 23] = "INVALID_TAG", e[e.INVALID_TAG_NAME = 25] = "INVALID_TAG_NAME", e[e.UNMATCHED_CLOSING_TAG = 26] = "UNMATCHED_CLOSING_TAG", e[e.UNCLOSED_TAG = 27] = "UNCLOSED_TAG";
})(R || (R = {}));
var Y;
(function(e) {
  e[e.literal = 0] = "literal", e[e.argument = 1] = "argument", e[e.number = 2] = "number", e[e.date = 3] = "date", e[e.time = 4] = "time", e[e.select = 5] = "select", e[e.plural = 6] = "plural", e[e.pound = 7] = "pound", e[e.tag = 8] = "tag";
})(Y || (Y = {}));
var wt;
(function(e) {
  e[e.number = 0] = "number", e[e.dateTime = 1] = "dateTime";
})(wt || (wt = {}));
function In(e) {
  return e.type === Y.literal;
}
function ys(e) {
  return e.type === Y.argument;
}
function pi(e) {
  return e.type === Y.number;
}
function mi(e) {
  return e.type === Y.date;
}
function gi(e) {
  return e.type === Y.time;
}
function vi(e) {
  return e.type === Y.select;
}
function bi(e) {
  return e.type === Y.plural;
}
function xs(e) {
  return e.type === Y.pound;
}
function _i(e) {
  return e.type === Y.tag;
}
function yi(e) {
  return !!(e && typeof e == "object" && e.type === wt.number);
}
function Wr(e) {
  return !!(e && typeof e == "object" && e.type === wt.dateTime);
}
var xi = /[ \xA0\u1680\u2000-\u200A\u202F\u205F\u3000]/, Es = /(?:[Eec]{1,6}|G{1,5}|[Qq]{1,5}|(?:[yYur]+|U{1,5})|[ML]{1,5}|d{1,2}|D{1,3}|F{1}|[abB]{1,5}|[hkHK]{1,2}|w{1,2}|W{1}|m{1,2}|s{1,2}|[zZOvVxX]{1,4})(?=([^']*'[^']*')*[^']*$)/g;
function ws(e) {
  var t = {};
  return e.replace(Es, function(r) {
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
var Ts = /[\t-\r \x85\u200E\u200F\u2028\u2029]/i;
function Ss(e) {
  if (e.length === 0)
    throw new Error("Number skeleton cannot be empty");
  for (var t = e.split(Ts).filter(function(_) {
    return _.length > 0;
  }), r = [], n = 0, i = t; n < i.length; n++) {
    var a = i[n], o = a.split("/");
    if (o.length === 0)
      throw new Error("Invalid number skeleton");
    for (var s = o[0], c = o.slice(1), f = 0, h = c; f < h.length; f++) {
      var p = h[f];
      if (p.length === 0)
        throw new Error("Invalid number skeleton");
    }
    r.push({ stem: s, options: c });
  }
  return r;
}
function As(e) {
  return e.replace(/^(.*?)-/, "");
}
var On = /^\.(?:(0+)(\*)?|(#+)|(0+)(#+))$/g, Ei = /^(@+)?(\+|#+)?[rs]?$/g, Hs = /(\*)(0+)|(#+)(0+)|(0+)/g, wi = /^(0+)$/;
function Bn(e) {
  var t = {};
  return e[e.length - 1] === "r" ? t.roundingPriority = "morePrecision" : e[e.length - 1] === "s" && (t.roundingPriority = "lessPrecision"), e.replace(Ei, function(r, n, i) {
    return typeof i != "string" ? (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length) : i === "+" ? t.minimumSignificantDigits = n.length : n[0] === "#" ? t.maximumSignificantDigits = n.length : (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length + (typeof i == "string" ? i.length : 0)), "";
  }), t;
}
function Ti(e) {
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
function Ps(e) {
  var t;
  if (e[0] === "E" && e[1] === "E" ? (t = {
    notation: "engineering"
  }, e = e.slice(2)) : e[0] === "E" && (t = {
    notation: "scientific"
  }, e = e.slice(1)), t) {
    var r = e.slice(0, 2);
    if (r === "+!" ? (t.signDisplay = "always", e = e.slice(2)) : r === "+?" && (t.signDisplay = "exceptZero", e = e.slice(2)), !wi.test(e))
      throw new Error("Malformed concise eng/scientific notation");
    t.minimumIntegerDigits = e.length;
  }
  return t;
}
function Mn(e) {
  var t = {}, r = Ti(e);
  return r || t;
}
function Is(e) {
  for (var t = {}, r = 0, n = e; r < n.length; r++) {
    var i = n[r];
    switch (i.stem) {
      case "percent":
      case "%":
        t.style = "percent";
        continue;
      case "%x100":
        t.style = "percent", t.scale = 100;
        continue;
      case "currency":
        t.style = "currency", t.currency = i.options[0];
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
        t.style = "unit", t.unit = As(i.options[0]);
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
        t = F(F(F({}, t), { notation: "scientific" }), i.options.reduce(function(c, f) {
          return F(F({}, c), Mn(f));
        }, {}));
        continue;
      case "engineering":
        t = F(F(F({}, t), { notation: "engineering" }), i.options.reduce(function(c, f) {
          return F(F({}, c), Mn(f));
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
        t.scale = parseFloat(i.options[0]);
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
        if (i.options.length > 1)
          throw new RangeError("integer-width stems only accept a single optional option");
        i.options[0].replace(Hs, function(c, f, h, p, _, w) {
          if (f)
            t.minimumIntegerDigits = h.length;
          else {
            if (p && _)
              throw new Error("We currently do not support maximum integer digits");
            if (w)
              throw new Error("We currently do not support exact integer digits");
          }
          return "";
        });
        continue;
    }
    if (wi.test(i.stem)) {
      t.minimumIntegerDigits = i.stem.length;
      continue;
    }
    if (On.test(i.stem)) {
      if (i.options.length > 1)
        throw new RangeError("Fraction-precision stems only accept a single optional option");
      i.stem.replace(On, function(c, f, h, p, _, w) {
        return h === "*" ? t.minimumFractionDigits = f.length : p && p[0] === "#" ? t.maximumFractionDigits = p.length : _ && w ? (t.minimumFractionDigits = _.length, t.maximumFractionDigits = _.length + w.length) : (t.minimumFractionDigits = f.length, t.maximumFractionDigits = f.length), "";
      });
      var a = i.options[0];
      a === "w" ? t = F(F({}, t), { trailingZeroDisplay: "stripIfInteger" }) : a && (t = F(F({}, t), Bn(a)));
      continue;
    }
    if (Ei.test(i.stem)) {
      t = F(F({}, t), Bn(i.stem));
      continue;
    }
    var o = Ti(i.stem);
    o && (t = F(F({}, t), o));
    var s = Ps(i.stem);
    s && (t = F(F({}, t), s));
  }
  return t;
}
var $t = {
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
function Os(e, t) {
  for (var r = "", n = 0; n < e.length; n++) {
    var i = e.charAt(n);
    if (i === "j") {
      for (var a = 0; n + 1 < e.length && e.charAt(n + 1) === i; )
        a++, n++;
      var o = 1 + (a & 1), s = a < 2 ? 1 : 3 + (a >> 1), c = "a", f = Bs(t);
      for ((f == "H" || f == "k") && (s = 0); s-- > 0; )
        r += c;
      for (; o-- > 0; )
        r = f + r;
    } else i === "J" ? r += "H" : r += i;
  }
  return r;
}
function Bs(e) {
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
  var i = $t[n || ""] || $t[r || ""] || $t["".concat(r, "-001")] || $t["001"];
  return i[0];
}
var Nr, Ms = new RegExp("^".concat(xi.source, "*")), Ls = new RegExp("".concat(xi.source, "*$"));
function k(e, t) {
  return { start: e, end: t };
}
var Ns = !!String.prototype.startsWith && "_a".startsWith("a", 1), Cs = !!String.fromCodePoint, Rs = !!Object.fromEntries, Ds = !!String.prototype.codePointAt, ks = !!String.prototype.trimStart, Us = !!String.prototype.trimEnd, Gs = !!Number.isSafeInteger, Fs = Gs ? Number.isSafeInteger : function(e) {
  return typeof e == "number" && isFinite(e) && Math.floor(e) === e && Math.abs(e) <= 9007199254740991;
}, Zr = !0;
try {
  var js = Ai("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  Zr = ((Nr = js.exec("a")) === null || Nr === void 0 ? void 0 : Nr[0]) === "a";
} catch {
  Zr = !1;
}
var Ln = Ns ? (
  // Native
  function(t, r, n) {
    return t.startsWith(r, n);
  }
) : (
  // For IE11
  function(t, r, n) {
    return t.slice(n, n + r.length) === r;
  }
), Yr = Cs ? String.fromCodePoint : (
  // IE11
  function() {
    for (var t = [], r = 0; r < arguments.length; r++)
      t[r] = arguments[r];
    for (var n = "", i = t.length, a = 0, o; i > a; ) {
      if (o = t[a++], o > 1114111)
        throw RangeError(o + " is not a valid code point");
      n += o < 65536 ? String.fromCharCode(o) : String.fromCharCode(((o -= 65536) >> 10) + 55296, o % 1024 + 56320);
    }
    return n;
  }
), Nn = (
  // native
  Rs ? Object.fromEntries : (
    // Ponyfill
    function(t) {
      for (var r = {}, n = 0, i = t; n < i.length; n++) {
        var a = i[n], o = a[0], s = a[1];
        r[o] = s;
      }
      return r;
    }
  )
), Si = Ds ? (
  // Native
  function(t, r) {
    return t.codePointAt(r);
  }
) : (
  // IE 11
  function(t, r) {
    var n = t.length;
    if (!(r < 0 || r >= n)) {
      var i = t.charCodeAt(r), a;
      return i < 55296 || i > 56319 || r + 1 === n || (a = t.charCodeAt(r + 1)) < 56320 || a > 57343 ? i : (i - 55296 << 10) + (a - 56320) + 65536;
    }
  }
), Vs = ks ? (
  // Native
  function(t) {
    return t.trimStart();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(Ms, "");
  }
), zs = Us ? (
  // Native
  function(t) {
    return t.trimEnd();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(Ls, "");
  }
);
function Ai(e, t) {
  return new RegExp(e, t);
}
var Jr;
if (Zr) {
  var Cn = Ai("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  Jr = function(t, r) {
    var n;
    Cn.lastIndex = r;
    var i = Cn.exec(t);
    return (n = i[1]) !== null && n !== void 0 ? n : "";
  };
} else
  Jr = function(t, r) {
    for (var n = []; ; ) {
      var i = Si(t, r);
      if (i === void 0 || Hi(i) || Zs(i))
        break;
      n.push(i), r += i >= 65536 ? 2 : 1;
    }
    return Yr.apply(void 0, n);
  };
var Xs = (
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
      for (var i = []; !this.isEOF(); ) {
        var a = this.char();
        if (a === 123) {
          var o = this.parseArgument(t, n);
          if (o.err)
            return o;
          i.push(o.val);
        } else {
          if (a === 125 && t > 0)
            break;
          if (a === 35 && (r === "plural" || r === "selectordinal")) {
            var s = this.clonePosition();
            this.bump(), i.push({
              type: Y.pound,
              location: k(s, this.clonePosition())
            });
          } else if (a === 60 && !this.ignoreTag && this.peek() === 47) {
            if (n)
              break;
            return this.error(R.UNMATCHED_CLOSING_TAG, k(this.clonePosition(), this.clonePosition()));
          } else if (a === 60 && !this.ignoreTag && Qr(this.peek() || 0)) {
            var o = this.parseTag(t, r);
            if (o.err)
              return o;
            i.push(o.val);
          } else {
            var o = this.parseLiteral(t, r);
            if (o.err)
              return o;
            i.push(o.val);
          }
        }
      }
      return { val: i, err: null };
    }, e.prototype.parseTag = function(t, r) {
      var n = this.clonePosition();
      this.bump();
      var i = this.parseTagName();
      if (this.bumpSpace(), this.bumpIf("/>"))
        return {
          val: {
            type: Y.literal,
            value: "<".concat(i, "/>"),
            location: k(n, this.clonePosition())
          },
          err: null
        };
      if (this.bumpIf(">")) {
        var a = this.parseMessage(t + 1, r, !0);
        if (a.err)
          return a;
        var o = a.val, s = this.clonePosition();
        if (this.bumpIf("</")) {
          if (this.isEOF() || !Qr(this.char()))
            return this.error(R.INVALID_TAG, k(s, this.clonePosition()));
          var c = this.clonePosition(), f = this.parseTagName();
          return i !== f ? this.error(R.UNMATCHED_CLOSING_TAG, k(c, this.clonePosition())) : (this.bumpSpace(), this.bumpIf(">") ? {
            val: {
              type: Y.tag,
              value: i,
              children: o,
              location: k(n, this.clonePosition())
            },
            err: null
          } : this.error(R.INVALID_TAG, k(s, this.clonePosition())));
        } else
          return this.error(R.UNCLOSED_TAG, k(n, this.clonePosition()));
      } else
        return this.error(R.INVALID_TAG, k(n, this.clonePosition()));
    }, e.prototype.parseTagName = function() {
      var t = this.offset();
      for (this.bump(); !this.isEOF() && Ws(this.char()); )
        this.bump();
      return this.message.slice(t, this.offset());
    }, e.prototype.parseLiteral = function(t, r) {
      for (var n = this.clonePosition(), i = ""; ; ) {
        var a = this.tryParseQuote(r);
        if (a) {
          i += a;
          continue;
        }
        var o = this.tryParseUnquoted(t, r);
        if (o) {
          i += o;
          continue;
        }
        var s = this.tryParseLeftAngleBracket();
        if (s) {
          i += s;
          continue;
        }
        break;
      }
      var c = k(n, this.clonePosition());
      return {
        val: { type: Y.literal, value: i, location: c },
        err: null
      };
    }, e.prototype.tryParseLeftAngleBracket = function() {
      return !this.isEOF() && this.char() === 60 && (this.ignoreTag || // If at the opening tag or closing tag position, bail.
      !qs(this.peek() || 0)) ? (this.bump(), "<") : null;
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
      return Yr.apply(void 0, r);
    }, e.prototype.tryParseUnquoted = function(t, r) {
      if (this.isEOF())
        return null;
      var n = this.char();
      return n === 60 || n === 123 || n === 35 && (r === "plural" || r === "selectordinal") || n === 125 && t > 0 ? null : (this.bump(), Yr(n));
    }, e.prototype.parseArgument = function(t, r) {
      var n = this.clonePosition();
      if (this.bump(), this.bumpSpace(), this.isEOF())
        return this.error(R.EXPECT_ARGUMENT_CLOSING_BRACE, k(n, this.clonePosition()));
      if (this.char() === 125)
        return this.bump(), this.error(R.EMPTY_ARGUMENT, k(n, this.clonePosition()));
      var i = this.parseIdentifierIfPossible().value;
      if (!i)
        return this.error(R.MALFORMED_ARGUMENT, k(n, this.clonePosition()));
      if (this.bumpSpace(), this.isEOF())
        return this.error(R.EXPECT_ARGUMENT_CLOSING_BRACE, k(n, this.clonePosition()));
      switch (this.char()) {
        // Simple argument: `{name}`
        case 125:
          return this.bump(), {
            val: {
              type: Y.argument,
              // value does not include the opening and closing braces.
              value: i,
              location: k(n, this.clonePosition())
            },
            err: null
          };
        // Argument with options: `{name, format, ...}`
        case 44:
          return this.bump(), this.bumpSpace(), this.isEOF() ? this.error(R.EXPECT_ARGUMENT_CLOSING_BRACE, k(n, this.clonePosition())) : this.parseArgumentOptions(t, r, i, n);
        default:
          return this.error(R.MALFORMED_ARGUMENT, k(n, this.clonePosition()));
      }
    }, e.prototype.parseIdentifierIfPossible = function() {
      var t = this.clonePosition(), r = this.offset(), n = Jr(this.message, r), i = r + n.length;
      this.bumpTo(i);
      var a = this.clonePosition(), o = k(t, a);
      return { value: n, location: o };
    }, e.prototype.parseArgumentOptions = function(t, r, n, i) {
      var a, o = this.clonePosition(), s = this.parseIdentifierIfPossible().value, c = this.clonePosition();
      switch (s) {
        case "":
          return this.error(R.EXPECT_ARGUMENT_TYPE, k(o, c));
        case "number":
        case "date":
        case "time": {
          this.bumpSpace();
          var f = null;
          if (this.bumpIf(",")) {
            this.bumpSpace();
            var h = this.clonePosition(), p = this.parseSimpleArgStyleIfPossible();
            if (p.err)
              return p;
            var _ = zs(p.val);
            if (_.length === 0)
              return this.error(R.EXPECT_ARGUMENT_STYLE, k(this.clonePosition(), this.clonePosition()));
            var w = k(h, this.clonePosition());
            f = { style: _, styleLocation: w };
          }
          var m = this.tryParseArgumentClose(i);
          if (m.err)
            return m;
          var T = k(i, this.clonePosition());
          if (f && Ln(f?.style, "::", 0)) {
            var M = Vs(f.style.slice(2));
            if (s === "number") {
              var p = this.parseNumberSkeletonFromString(M, f.styleLocation);
              return p.err ? p : {
                val: { type: Y.number, value: n, location: T, style: p.val },
                err: null
              };
            } else {
              if (M.length === 0)
                return this.error(R.EXPECT_DATE_TIME_SKELETON, T);
              var v = M;
              this.locale && (v = Os(M, this.locale));
              var _ = {
                type: wt.dateTime,
                pattern: v,
                location: f.styleLocation,
                parsedOptions: this.shouldParseSkeletons ? ws(v) : {}
              }, b = s === "date" ? Y.date : Y.time;
              return {
                val: { type: b, value: n, location: T, style: _ },
                err: null
              };
            }
          }
          return {
            val: {
              type: s === "number" ? Y.number : s === "date" ? Y.date : Y.time,
              value: n,
              location: T,
              style: (a = f?.style) !== null && a !== void 0 ? a : null
            },
            err: null
          };
        }
        case "plural":
        case "selectordinal":
        case "select": {
          var S = this.clonePosition();
          if (this.bumpSpace(), !this.bumpIf(","))
            return this.error(R.EXPECT_SELECT_ARGUMENT_OPTIONS, k(S, F({}, S)));
          this.bumpSpace();
          var y = this.parseIdentifierIfPossible(), E = 0;
          if (s !== "select" && y.value === "offset") {
            if (!this.bumpIf(":"))
              return this.error(R.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, k(this.clonePosition(), this.clonePosition()));
            this.bumpSpace();
            var p = this.tryParseDecimalInteger(R.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, R.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE);
            if (p.err)
              return p;
            this.bumpSpace(), y = this.parseIdentifierIfPossible(), E = p.val;
          }
          var O = this.tryParsePluralOrSelectOptions(t, s, r, y);
          if (O.err)
            return O;
          var m = this.tryParseArgumentClose(i);
          if (m.err)
            return m;
          var P = k(i, this.clonePosition());
          return s === "select" ? {
            val: {
              type: Y.select,
              value: n,
              options: Nn(O.val),
              location: P
            },
            err: null
          } : {
            val: {
              type: Y.plural,
              value: n,
              options: Nn(O.val),
              offset: E,
              pluralType: s === "plural" ? "cardinal" : "ordinal",
              location: P
            },
            err: null
          };
        }
        default:
          return this.error(R.INVALID_ARGUMENT_TYPE, k(o, c));
      }
    }, e.prototype.tryParseArgumentClose = function(t) {
      return this.isEOF() || this.char() !== 125 ? this.error(R.EXPECT_ARGUMENT_CLOSING_BRACE, k(t, this.clonePosition())) : (this.bump(), { val: !0, err: null });
    }, e.prototype.parseSimpleArgStyleIfPossible = function() {
      for (var t = 0, r = this.clonePosition(); !this.isEOF(); ) {
        var n = this.char();
        switch (n) {
          case 39: {
            this.bump();
            var i = this.clonePosition();
            if (!this.bumpUntil("'"))
              return this.error(R.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE, k(i, this.clonePosition()));
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
        n = Ss(t);
      } catch {
        return this.error(R.INVALID_NUMBER_SKELETON, r);
      }
      return {
        val: {
          type: wt.number,
          tokens: n,
          location: r,
          parsedOptions: this.shouldParseSkeletons ? Is(n) : {}
        },
        err: null
      };
    }, e.prototype.tryParsePluralOrSelectOptions = function(t, r, n, i) {
      for (var a, o = !1, s = [], c = /* @__PURE__ */ new Set(), f = i.value, h = i.location; ; ) {
        if (f.length === 0) {
          var p = this.clonePosition();
          if (r !== "select" && this.bumpIf("=")) {
            var _ = this.tryParseDecimalInteger(R.EXPECT_PLURAL_ARGUMENT_SELECTOR, R.INVALID_PLURAL_ARGUMENT_SELECTOR);
            if (_.err)
              return _;
            h = k(p, this.clonePosition()), f = this.message.slice(p.offset, this.offset());
          } else
            break;
        }
        if (c.has(f))
          return this.error(r === "select" ? R.DUPLICATE_SELECT_ARGUMENT_SELECTOR : R.DUPLICATE_PLURAL_ARGUMENT_SELECTOR, h);
        f === "other" && (o = !0), this.bumpSpace();
        var w = this.clonePosition();
        if (!this.bumpIf("{"))
          return this.error(r === "select" ? R.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT : R.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT, k(this.clonePosition(), this.clonePosition()));
        var m = this.parseMessage(t + 1, r, n);
        if (m.err)
          return m;
        var T = this.tryParseArgumentClose(w);
        if (T.err)
          return T;
        s.push([
          f,
          {
            value: m.val,
            location: k(w, this.clonePosition())
          }
        ]), c.add(f), this.bumpSpace(), a = this.parseIdentifierIfPossible(), f = a.value, h = a.location;
      }
      return s.length === 0 ? this.error(r === "select" ? R.EXPECT_SELECT_ARGUMENT_SELECTOR : R.EXPECT_PLURAL_ARGUMENT_SELECTOR, k(this.clonePosition(), this.clonePosition())) : this.requiresOtherClause && !o ? this.error(R.MISSING_OTHER_CLAUSE, k(this.clonePosition(), this.clonePosition())) : { val: s, err: null };
    }, e.prototype.tryParseDecimalInteger = function(t, r) {
      var n = 1, i = this.clonePosition();
      this.bumpIf("+") || this.bumpIf("-") && (n = -1);
      for (var a = !1, o = 0; !this.isEOF(); ) {
        var s = this.char();
        if (s >= 48 && s <= 57)
          a = !0, o = o * 10 + (s - 48), this.bump();
        else
          break;
      }
      var c = k(i, this.clonePosition());
      return a ? (o *= n, Fs(o) ? { val: o, err: null } : this.error(r, c)) : this.error(t, c);
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
      var r = Si(this.message, t);
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
      if (Ln(this.message, t, this.offset())) {
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
      for (; !this.isEOF() && Hi(this.char()); )
        this.bump();
    }, e.prototype.peek = function() {
      if (this.isEOF())
        return null;
      var t = this.char(), r = this.offset(), n = this.message.charCodeAt(r + (t >= 65536 ? 2 : 1));
      return n ?? null;
    }, e;
  })()
);
function Qr(e) {
  return e >= 97 && e <= 122 || e >= 65 && e <= 90;
}
function qs(e) {
  return Qr(e) || e === 47;
}
function Ws(e) {
  return e === 45 || e === 46 || e >= 48 && e <= 57 || e === 95 || e >= 97 && e <= 122 || e >= 65 && e <= 90 || e == 183 || e >= 192 && e <= 214 || e >= 216 && e <= 246 || e >= 248 && e <= 893 || e >= 895 && e <= 8191 || e >= 8204 && e <= 8205 || e >= 8255 && e <= 8256 || e >= 8304 && e <= 8591 || e >= 11264 && e <= 12271 || e >= 12289 && e <= 55295 || e >= 63744 && e <= 64975 || e >= 65008 && e <= 65533 || e >= 65536 && e <= 983039;
}
function Hi(e) {
  return e >= 9 && e <= 13 || e === 32 || e === 133 || e >= 8206 && e <= 8207 || e === 8232 || e === 8233;
}
function Zs(e) {
  return e >= 33 && e <= 35 || e === 36 || e >= 37 && e <= 39 || e === 40 || e === 41 || e === 42 || e === 43 || e === 44 || e === 45 || e >= 46 && e <= 47 || e >= 58 && e <= 59 || e >= 60 && e <= 62 || e >= 63 && e <= 64 || e === 91 || e === 92 || e === 93 || e === 94 || e === 96 || e === 123 || e === 124 || e === 125 || e === 126 || e === 161 || e >= 162 && e <= 165 || e === 166 || e === 167 || e === 169 || e === 171 || e === 172 || e === 174 || e === 176 || e === 177 || e === 182 || e === 187 || e === 191 || e === 215 || e === 247 || e >= 8208 && e <= 8213 || e >= 8214 && e <= 8215 || e === 8216 || e === 8217 || e === 8218 || e >= 8219 && e <= 8220 || e === 8221 || e === 8222 || e === 8223 || e >= 8224 && e <= 8231 || e >= 8240 && e <= 8248 || e === 8249 || e === 8250 || e >= 8251 && e <= 8254 || e >= 8257 && e <= 8259 || e === 8260 || e === 8261 || e === 8262 || e >= 8263 && e <= 8273 || e === 8274 || e === 8275 || e >= 8277 && e <= 8286 || e >= 8592 && e <= 8596 || e >= 8597 && e <= 8601 || e >= 8602 && e <= 8603 || e >= 8604 && e <= 8607 || e === 8608 || e >= 8609 && e <= 8610 || e === 8611 || e >= 8612 && e <= 8613 || e === 8614 || e >= 8615 && e <= 8621 || e === 8622 || e >= 8623 && e <= 8653 || e >= 8654 && e <= 8655 || e >= 8656 && e <= 8657 || e === 8658 || e === 8659 || e === 8660 || e >= 8661 && e <= 8691 || e >= 8692 && e <= 8959 || e >= 8960 && e <= 8967 || e === 8968 || e === 8969 || e === 8970 || e === 8971 || e >= 8972 && e <= 8991 || e >= 8992 && e <= 8993 || e >= 8994 && e <= 9e3 || e === 9001 || e === 9002 || e >= 9003 && e <= 9083 || e === 9084 || e >= 9085 && e <= 9114 || e >= 9115 && e <= 9139 || e >= 9140 && e <= 9179 || e >= 9180 && e <= 9185 || e >= 9186 && e <= 9254 || e >= 9255 && e <= 9279 || e >= 9280 && e <= 9290 || e >= 9291 && e <= 9311 || e >= 9472 && e <= 9654 || e === 9655 || e >= 9656 && e <= 9664 || e === 9665 || e >= 9666 && e <= 9719 || e >= 9720 && e <= 9727 || e >= 9728 && e <= 9838 || e === 9839 || e >= 9840 && e <= 10087 || e === 10088 || e === 10089 || e === 10090 || e === 10091 || e === 10092 || e === 10093 || e === 10094 || e === 10095 || e === 10096 || e === 10097 || e === 10098 || e === 10099 || e === 10100 || e === 10101 || e >= 10132 && e <= 10175 || e >= 10176 && e <= 10180 || e === 10181 || e === 10182 || e >= 10183 && e <= 10213 || e === 10214 || e === 10215 || e === 10216 || e === 10217 || e === 10218 || e === 10219 || e === 10220 || e === 10221 || e === 10222 || e === 10223 || e >= 10224 && e <= 10239 || e >= 10240 && e <= 10495 || e >= 10496 && e <= 10626 || e === 10627 || e === 10628 || e === 10629 || e === 10630 || e === 10631 || e === 10632 || e === 10633 || e === 10634 || e === 10635 || e === 10636 || e === 10637 || e === 10638 || e === 10639 || e === 10640 || e === 10641 || e === 10642 || e === 10643 || e === 10644 || e === 10645 || e === 10646 || e === 10647 || e === 10648 || e >= 10649 && e <= 10711 || e === 10712 || e === 10713 || e === 10714 || e === 10715 || e >= 10716 && e <= 10747 || e === 10748 || e === 10749 || e >= 10750 && e <= 11007 || e >= 11008 && e <= 11055 || e >= 11056 && e <= 11076 || e >= 11077 && e <= 11078 || e >= 11079 && e <= 11084 || e >= 11085 && e <= 11123 || e >= 11124 && e <= 11125 || e >= 11126 && e <= 11157 || e === 11158 || e >= 11159 && e <= 11263 || e >= 11776 && e <= 11777 || e === 11778 || e === 11779 || e === 11780 || e === 11781 || e >= 11782 && e <= 11784 || e === 11785 || e === 11786 || e === 11787 || e === 11788 || e === 11789 || e >= 11790 && e <= 11798 || e === 11799 || e >= 11800 && e <= 11801 || e === 11802 || e === 11803 || e === 11804 || e === 11805 || e >= 11806 && e <= 11807 || e === 11808 || e === 11809 || e === 11810 || e === 11811 || e === 11812 || e === 11813 || e === 11814 || e === 11815 || e === 11816 || e === 11817 || e >= 11818 && e <= 11822 || e === 11823 || e >= 11824 && e <= 11833 || e >= 11834 && e <= 11835 || e >= 11836 && e <= 11839 || e === 11840 || e === 11841 || e === 11842 || e >= 11843 && e <= 11855 || e >= 11856 && e <= 11857 || e === 11858 || e >= 11859 && e <= 11903 || e >= 12289 && e <= 12291 || e === 12296 || e === 12297 || e === 12298 || e === 12299 || e === 12300 || e === 12301 || e === 12302 || e === 12303 || e === 12304 || e === 12305 || e >= 12306 && e <= 12307 || e === 12308 || e === 12309 || e === 12310 || e === 12311 || e === 12312 || e === 12313 || e === 12314 || e === 12315 || e === 12316 || e === 12317 || e >= 12318 && e <= 12319 || e === 12320 || e === 12336 || e === 64830 || e === 64831 || e >= 65093 && e <= 65094;
}
function Kr(e) {
  e.forEach(function(t) {
    if (delete t.location, vi(t) || bi(t))
      for (var r in t.options)
        delete t.options[r].location, Kr(t.options[r].value);
    else pi(t) && yi(t.style) || (mi(t) || gi(t)) && Wr(t.style) ? delete t.style.location : _i(t) && Kr(t.children);
  });
}
function Ys(e, t) {
  t === void 0 && (t = {}), t = F({ shouldParseSkeletons: !0, requiresOtherClause: !0 }, t);
  var r = new Xs(e, t).parse();
  if (r.err) {
    var n = SyntaxError(R[r.err.kind]);
    throw n.location = r.err.location, n.originalMessage = r.err.message, n;
  }
  return t?.captureLocation || Kr(r.val), r.val;
}
var Tt;
(function(e) {
  e.MISSING_VALUE = "MISSING_VALUE", e.INVALID_VALUE = "INVALID_VALUE", e.MISSING_INTL_API = "MISSING_INTL_API";
})(Tt || (Tt = {}));
var mr = (
  /** @class */
  (function(e) {
    pr(t, e);
    function t(r, n, i) {
      var a = e.call(this, r) || this;
      return a.code = n, a.originalMessage = i, a;
    }
    return t.prototype.toString = function() {
      return "[formatjs Error: ".concat(this.code, "] ").concat(this.message);
    }, t;
  })(Error)
), Rn = (
  /** @class */
  (function(e) {
    pr(t, e);
    function t(r, n, i, a) {
      return e.call(this, 'Invalid values for "'.concat(r, '": "').concat(n, '". Options are "').concat(Object.keys(i).join('", "'), '"'), Tt.INVALID_VALUE, a) || this;
    }
    return t;
  })(mr)
), Js = (
  /** @class */
  (function(e) {
    pr(t, e);
    function t(r, n, i) {
      return e.call(this, 'Value for "'.concat(r, '" must be of type ').concat(n), Tt.INVALID_VALUE, i) || this;
    }
    return t;
  })(mr)
), Qs = (
  /** @class */
  (function(e) {
    pr(t, e);
    function t(r, n) {
      return e.call(this, 'The intl string context variable "'.concat(r, '" was not provided to the string "').concat(n, '"'), Tt.MISSING_VALUE, n) || this;
    }
    return t;
  })(mr)
), ye;
(function(e) {
  e[e.literal = 0] = "literal", e[e.object = 1] = "object";
})(ye || (ye = {}));
function Ks(e) {
  return e.length < 2 ? e : e.reduce(function(t, r) {
    var n = t[t.length - 1];
    return !n || n.type !== ye.literal || r.type !== ye.literal ? t.push(r) : n.value += r.value, t;
  }, []);
}
function $s(e) {
  return typeof e == "function";
}
function rr(e, t, r, n, i, a, o) {
  if (e.length === 1 && In(e[0]))
    return [
      {
        type: ye.literal,
        value: e[0].value
      }
    ];
  for (var s = [], c = 0, f = e; c < f.length; c++) {
    var h = f[c];
    if (In(h)) {
      s.push({
        type: ye.literal,
        value: h.value
      });
      continue;
    }
    if (xs(h)) {
      typeof a == "number" && s.push({
        type: ye.literal,
        value: r.getNumberFormat(t).format(a)
      });
      continue;
    }
    var p = h.value;
    if (!(i && p in i))
      throw new Qs(p, o);
    var _ = i[p];
    if (ys(h)) {
      (!_ || typeof _ == "string" || typeof _ == "number") && (_ = typeof _ == "string" || typeof _ == "number" ? String(_) : ""), s.push({
        type: typeof _ == "string" ? ye.literal : ye.object,
        value: _
      });
      continue;
    }
    if (mi(h)) {
      var w = typeof h.style == "string" ? n.date[h.style] : Wr(h.style) ? h.style.parsedOptions : void 0;
      s.push({
        type: ye.literal,
        value: r.getDateTimeFormat(t, w).format(_)
      });
      continue;
    }
    if (gi(h)) {
      var w = typeof h.style == "string" ? n.time[h.style] : Wr(h.style) ? h.style.parsedOptions : n.time.medium;
      s.push({
        type: ye.literal,
        value: r.getDateTimeFormat(t, w).format(_)
      });
      continue;
    }
    if (pi(h)) {
      var w = typeof h.style == "string" ? n.number[h.style] : yi(h.style) ? h.style.parsedOptions : void 0;
      w && w.scale && (_ = _ * (w.scale || 1)), s.push({
        type: ye.literal,
        value: r.getNumberFormat(t, w).format(_)
      });
      continue;
    }
    if (_i(h)) {
      var m = h.children, T = h.value, M = i[T];
      if (!$s(M))
        throw new Js(T, "function", o);
      var v = rr(m, t, r, n, i, a), b = M(v.map(function(E) {
        return E.value;
      }));
      Array.isArray(b) || (b = [b]), s.push.apply(s, b.map(function(E) {
        return {
          type: typeof E == "string" ? ye.literal : ye.object,
          value: E
        };
      }));
    }
    if (vi(h)) {
      var S = h.options[_] || h.options.other;
      if (!S)
        throw new Rn(h.value, _, Object.keys(h.options), o);
      s.push.apply(s, rr(S.value, t, r, n, i));
      continue;
    }
    if (bi(h)) {
      var S = h.options["=".concat(_)];
      if (!S) {
        if (!Intl.PluralRules)
          throw new mr(`Intl.PluralRules is not available in this environment.
Try polyfilling it using "@formatjs/intl-pluralrules"
`, Tt.MISSING_INTL_API, o);
        var y = r.getPluralRules(t, { type: h.pluralType }).select(_ - (h.offset || 0));
        S = h.options[y] || h.options.other;
      }
      if (!S)
        throw new Rn(h.value, _, Object.keys(h.options), o);
      s.push.apply(s, rr(S.value, t, r, n, i, _ - (h.offset || 0)));
      continue;
    }
  }
  return Ks(s);
}
function eo(e, t) {
  return t ? F(F(F({}, e || {}), t || {}), Object.keys(e).reduce(function(r, n) {
    return r[n] = F(F({}, e[n]), t[n] || {}), r;
  }, {})) : e;
}
function to(e, t) {
  return t ? Object.keys(e).reduce(function(r, n) {
    return r[n] = eo(e[n], t[n]), r;
  }, F({}, e)) : e;
}
function Cr(e) {
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
function ro(e) {
  return e === void 0 && (e = {
    number: {},
    dateTime: {},
    pluralRules: {}
  }), {
    getNumberFormat: Mr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.NumberFormat).bind.apply(t, Br([void 0], r, !1)))();
    }, {
      cache: Cr(e.number),
      strategy: Lr.variadic
    }),
    getDateTimeFormat: Mr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.DateTimeFormat).bind.apply(t, Br([void 0], r, !1)))();
    }, {
      cache: Cr(e.dateTime),
      strategy: Lr.variadic
    }),
    getPluralRules: Mr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.PluralRules).bind.apply(t, Br([void 0], r, !1)))();
    }, {
      cache: Cr(e.pluralRules),
      strategy: Lr.variadic
    })
  };
}
var no = (
  /** @class */
  (function() {
    function e(t, r, n, i) {
      r === void 0 && (r = e.defaultLocale);
      var a = this;
      if (this.formatterCache = {
        number: {},
        dateTime: {},
        pluralRules: {}
      }, this.format = function(c) {
        var f = a.formatToParts(c);
        if (f.length === 1)
          return f[0].value;
        var h = f.reduce(function(p, _) {
          return !p.length || _.type !== ye.literal || typeof p[p.length - 1] != "string" ? p.push(_.value) : p[p.length - 1] += _.value, p;
        }, []);
        return h.length <= 1 ? h[0] || "" : h;
      }, this.formatToParts = function(c) {
        return rr(a.ast, a.locales, a.formatters, a.formats, c, void 0, a.message);
      }, this.resolvedOptions = function() {
        var c;
        return {
          locale: ((c = a.resolvedLocale) === null || c === void 0 ? void 0 : c.toString()) || Intl.NumberFormat.supportedLocalesOf(a.locales)[0]
        };
      }, this.getAst = function() {
        return a.ast;
      }, this.locales = r, this.resolvedLocale = e.resolveLocale(r), typeof t == "string") {
        if (this.message = t, !e.__parse)
          throw new TypeError("IntlMessageFormat.__parse must be set to process `message` of type `string`");
        var o = i || {};
        o.formatters;
        var s = hs(o, ["formatters"]);
        this.ast = e.__parse(t, F(F({}, s), { locale: this.resolvedLocale }));
      } else
        this.ast = t;
      if (!Array.isArray(this.ast))
        throw new TypeError("A message must be provided as a String or AST.");
      this.formats = to(e.formats, n), this.formatters = i && i.formatters || ro(this.formatterCache);
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
    }, e.__parse = Ys, e.formats = {
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
function io(e, t) {
  if (t == null)
    return;
  if (t in e)
    return e[t];
  const r = t.split(".");
  let n = e;
  for (let i = 0; i < r.length; i++)
    if (typeof n == "object") {
      if (i > 0) {
        const a = r.slice(i, r.length).join(".");
        if (a in n) {
          n = n[a];
          break;
        }
      }
      n = n[r[i]];
    } else
      n = void 0;
  return n;
}
const Ke = {}, ao = (e, t, r) => r && (t in Ke || (Ke[t] = {}), e in Ke[t] || (Ke[t][e] = r), r), Pi = (e, t) => {
  if (t == null)
    return;
  if (t in Ke && e in Ke[t])
    return Ke[t][e];
  const r = gr(t);
  for (let n = 0; n < r.length; n++) {
    const i = r[n], a = oo(i, e);
    if (a)
      return ao(e, t, a);
  }
};
let cn;
const Vt = jt({});
function so(e) {
  return cn[e] || null;
}
function Ii(e) {
  return e in cn;
}
function oo(e, t) {
  if (!Ii(e))
    return null;
  const r = so(e);
  return io(r, t);
}
function lo(e) {
  if (e == null)
    return;
  const t = gr(e);
  for (let r = 0; r < t.length; r++) {
    const n = t[r];
    if (Ii(n))
      return n;
  }
}
function uo(e, ...t) {
  delete Ke[e], Vt.update((r) => (r[e] = cs.all([r[e] || {}, ...t]), r));
}
At(
  [Vt],
  ([e]) => Object.keys(e)
);
Vt.subscribe((e) => cn = e);
const nr = {};
function fo(e, t) {
  nr[e].delete(t), nr[e].size === 0 && delete nr[e];
}
function Oi(e) {
  return nr[e];
}
function co(e) {
  return gr(e).map((t) => {
    const r = Oi(t);
    return [t, r ? [...r] : []];
  }).filter(([, t]) => t.length > 0);
}
function $r(e) {
  return e == null ? !1 : gr(e).some(
    (t) => {
      var r;
      return (r = Oi(t)) == null ? void 0 : r.size;
    }
  );
}
function ho(e, t) {
  return Promise.all(
    t.map((n) => (fo(e, n), n().then((i) => i.default || i)))
  ).then((n) => uo(e, ...n));
}
const Ct = {};
function Bi(e) {
  if (!$r(e))
    return e in Ct ? Ct[e] : Promise.resolve();
  const t = co(e);
  return Ct[e] = Promise.all(
    t.map(
      ([r, n]) => ho(r, n)
    )
  ).then(() => {
    if ($r(e))
      return Bi(e);
    delete Ct[e];
  }), Ct[e];
}
const po = {
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
}, mo = {
  fallbackLocale: null,
  loadingDelay: 200,
  formats: po,
  warnOnMissingMessages: !0,
  handleMissingMessage: void 0,
  ignoreTag: !0
}, go = mo;
function St() {
  return go;
}
const Rr = jt(!1);
var vo = Object.defineProperty, bo = Object.defineProperties, _o = Object.getOwnPropertyDescriptors, Dn = Object.getOwnPropertySymbols, yo = Object.prototype.hasOwnProperty, xo = Object.prototype.propertyIsEnumerable, kn = (e, t, r) => t in e ? vo(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, Eo = (e, t) => {
  for (var r in t || (t = {}))
    yo.call(t, r) && kn(e, r, t[r]);
  if (Dn)
    for (var r of Dn(t))
      xo.call(t, r) && kn(e, r, t[r]);
  return e;
}, wo = (e, t) => bo(e, _o(t));
let en;
const or = jt(null);
function Un(e) {
  return e.split("-").map((t, r, n) => n.slice(0, r + 1).join("-")).reverse();
}
function gr(e, t = St().fallbackLocale) {
  const r = Un(e);
  return t ? [.../* @__PURE__ */ new Set([...r, ...Un(t)])] : r;
}
function ht() {
  return en ?? void 0;
}
or.subscribe((e) => {
  en = e ?? void 0, typeof window < "u" && e != null && document.documentElement.setAttribute("lang", e);
});
const To = (e) => {
  if (e && lo(e) && $r(e)) {
    const { loadingDelay: t } = St();
    let r;
    return typeof window < "u" && ht() != null && t ? r = window.setTimeout(
      () => Rr.set(!0),
      t
    ) : Rr.set(!0), Bi(e).then(() => {
      or.set(e);
    }).finally(() => {
      clearTimeout(r), Rr.set(!1);
    });
  }
  return or.set(e);
}, Ht = wo(Eo({}, or), {
  set: To
}), vr = (e) => {
  const t = /* @__PURE__ */ Object.create(null);
  return (n) => {
    const i = JSON.stringify(n);
    return i in t ? t[i] : t[i] = e(n);
  };
};
var So = Object.defineProperty, lr = Object.getOwnPropertySymbols, Mi = Object.prototype.hasOwnProperty, Li = Object.prototype.propertyIsEnumerable, Gn = (e, t, r) => t in e ? So(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, hn = (e, t) => {
  for (var r in t || (t = {}))
    Mi.call(t, r) && Gn(e, r, t[r]);
  if (lr)
    for (var r of lr(t))
      Li.call(t, r) && Gn(e, r, t[r]);
  return e;
}, Pt = (e, t) => {
  var r = {};
  for (var n in e)
    Mi.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && lr)
    for (var n of lr(e))
      t.indexOf(n) < 0 && Li.call(e, n) && (r[n] = e[n]);
  return r;
};
const Ut = (e, t) => {
  const { formats: r } = St();
  if (e in r && t in r[e])
    return r[e][t];
  throw new Error(`[svelte-i18n] Unknown "${t}" ${e} format.`);
}, Ao = vr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = Pt(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format numbers');
    return n && (i = Ut("number", n)), new Intl.NumberFormat(r, i);
  }
), Ho = vr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = Pt(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format dates');
    return n ? i = Ut("date", n) : Object.keys(i).length === 0 && (i = Ut("date", "short")), new Intl.DateTimeFormat(r, i);
  }
), Po = vr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = Pt(t, ["locale", "format"]);
    if (r == null)
      throw new Error(
        '[svelte-i18n] A "locale" must be set to format time values'
      );
    return n ? i = Ut("time", n) : Object.keys(i).length === 0 && (i = Ut("time", "short")), new Intl.DateTimeFormat(r, i);
  }
), Io = (e = {}) => {
  var t = e, {
    locale: r = ht()
  } = t, n = Pt(t, [
    "locale"
  ]);
  return Ao(hn({ locale: r }, n));
}, Oo = (e = {}) => {
  var t = e, {
    locale: r = ht()
  } = t, n = Pt(t, [
    "locale"
  ]);
  return Ho(hn({ locale: r }, n));
}, Bo = (e = {}) => {
  var t = e, {
    locale: r = ht()
  } = t, n = Pt(t, [
    "locale"
  ]);
  return Po(hn({ locale: r }, n));
}, Mo = vr(
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  (e, t = ht()) => new no(e, t, St().formats, {
    ignoreTag: St().ignoreTag
  })
), Lo = (e, t = {}) => {
  var r, n, i, a;
  let o = t;
  typeof e == "object" && (o = e, e = o.id);
  const {
    values: s,
    locale: c = ht(),
    default: f
  } = o;
  if (c == null)
    throw new Error(
      "[svelte-i18n] Cannot format a message without first setting the initial locale."
    );
  let h = Pi(e, c);
  if (!h)
    h = (a = (i = (n = (r = St()).handleMissingMessage) == null ? void 0 : n.call(r, { locale: c, id: e, defaultValue: f })) != null ? i : f) != null ? a : e;
  else if (typeof h != "string")
    return console.warn(
      `[svelte-i18n] Message with id "${e}" must be of type "string", found: "${typeof h}". Gettin its value through the "$format" method is deprecated; use the "json" method instead.`
    ), h;
  if (!s)
    return h;
  let p = h;
  try {
    p = Mo(h, c).format(s);
  } catch (_) {
    _ instanceof Error && console.warn(
      `[svelte-i18n] Message "${e}" has syntax error:`,
      _.message
    );
  }
  return p;
}, No = (e, t) => Bo(t).format(e), Co = (e, t) => Oo(t).format(e), Ro = (e, t) => Io(t).format(e), Do = (e, t = ht()) => Pi(e, t);
At([Ht, Vt], () => Lo);
At([Ht], () => No);
At([Ht], () => Co);
At([Ht], () => Ro);
At([Ht, Vt], () => Do);
const ko = "__i18n__", Uo = [
  "label",
  "info",
  "placeholder",
  "description",
  "title",
  "value"
], Go = [
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
function Fo(e) {
  return typeof e == "string" && e.includes(ko);
}
class jo {
  load_component;
  #t = V(kt({}));
  get shared() {
    return l(this.#t);
  }
  set shared(t) {
    x(this.#t, t, !0);
  }
  #r = V(kt({}));
  get props() {
    return l(this.#r);
  }
  set props(t) {
    x(this.#r, t, !0);
  }
  #e = V((t) => t);
  get i18n() {
    return l(this.#e);
  }
  set i18n(t) {
    x(this.#e, t, !0);
  }
  translatable_props = {};
  dispatcher;
  last_update = null;
  shared_props = Go;
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
    for (const n of Uo)
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
    ), Be(() => {
      for (const n in t.shared_props)
        this._is_i18n_managed(`shared.${n}`, t.shared_props[n]) || (this.shared[n] = t.shared_props[n]);
      for (const n in t.props)
        this._is_i18n_managed(`props.${n}`, t.props[n]) || (this.props[n] = t.props[n]);
      this.register_component(
        t.shared_props.id,
        // @ts-ignore
        this.set_data.bind(this),
        this.get_data.bind(this)
      ), oe(() => {
        this.shared.id = t.shared_props.id;
      });
    }), Object.keys(this.translatable_props).length > 0 && Ht.subscribe(() => {
      for (const [n, i] of Object.entries(this.translatable_props)) {
        const [a, o] = n.split("."), s = this.i18n(i);
        a === "shared" ? this.shared[o] = s : this.props[o] = s;
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
    const i = this.i18n(n);
    return i !== n && (this.translatable_props[`${t}.${r}`] = n), i;
  }
  dispatch(t, r) {
    this.dispatcher(this.shared.id, t, r);
  }
  async get_data() {
    return Sa(this.props);
  }
  update(t) {
    this.set_data(t);
  }
  set_data(t) {
    for (const r in t) {
      const n = t[r], i = Fo(n) ? this._translate_and_store(this.shared_props.includes(r) ? "shared" : "props", r, n) : n;
      if (this.shared_props.includes(r)) {
        const a = r;
        this.shared[a] = i;
        continue;
      }
      this.props[r] = i;
    }
  }
  watch_for_change() {
    Be(() => {
      this.mounted || (this.old_value = this.props.value, this.mounted = !0), this.old_value != this.props.value && (this.old_value = this.props.value, this.dispatch("change"));
    });
  }
}
ya();
var Vo = /* @__PURE__ */ ii('<svg class="resize-handle svelte-f3h6t9" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><line x1="1" y1="9" x2="9" y2="1" stroke="gray" stroke-width="0.5" class="svelte-f3h6t9"></line><line x1="5" y1="9" x2="9" y2="5" stroke="gray" stroke-width="0.5" class="svelte-f3h6t9"></line></svg>'), Fn = /* @__PURE__ */ de("<!> <!>", 1), zo = /* @__PURE__ */ de('<div class="placeholder svelte-f3h6t9"></div>');
function Xo(e, t) {
  hr(t, !1);
  let r = B(t, "height", 8, void 0), n = B(t, "min_height", 8, void 0), i = B(t, "max_height", 8, void 0), a = B(t, "width", 8, void 0), o = B(t, "elem_id", 8, ""), s = B(t, "elem_classes", 24, () => []), c = B(t, "variant", 8, "solid"), f = B(t, "border_mode", 8, "base"), h = B(t, "padding", 8, !0), p = B(t, "type", 8, "normal"), _ = B(t, "test_id", 8, void 0), w = B(t, "explicit_call", 8, !1), m = B(t, "container", 8, !0), T = B(t, "visible", 8, !0), M = B(t, "allow_overflow", 8, !0), v = B(t, "overflow_behavior", 8, "auto"), b = B(t, "scale", 8, null), S = B(t, "min_width", 8, 0), y = B(t, "flex", 12, !1), E = B(t, "resizable", 8, !1), O = B(t, "rtl", 8, !1), P = B(t, "fullscreen", 12, !1), I = B(t, "label", 8, void 0), U = lt(P()), j = lt(), ce = p() === "fieldset" ? "fieldset" : "div", Ee = lt(0), le = lt(0), X = lt(null);
  function Se(ne) {
    P() && ne.key === "Escape" && P(!1);
  }
  const xe = (ne) => {
    if (ne !== void 0) {
      if (typeof ne == "number")
        return ne + "px";
      if (typeof ne == "string")
        return ne;
    }
  }, Ae = (ne) => {
    let Me = ne.clientY;
    const pe = (K) => {
      const he = K.clientY - Me;
      Me = K.clientY, Ea(j, l(j).style.height = `${l(j).offsetHeight + he}px`);
    }, Le = () => {
      window.removeEventListener("mousemove", pe), window.removeEventListener("mouseup", Le);
    };
    window.addEventListener("mousemove", pe), window.addEventListener("mouseup", Le);
  };
  vn(
    () => (Oe(P()), l(U), l(j)),
    () => {
      P() !== l(U) && (x(U, P()), P() ? (x(X, l(j).getBoundingClientRect()), x(Ee, l(j).offsetHeight), x(le, l(j).offsetWidth), window.addEventListener("keydown", Se)) : (x(X, null), window.removeEventListener("keydown", Se)));
    }
  ), vn(() => Oe(T()), () => {
    T() || y(!1);
  }), xa(), rs();
  var je = _t(), ue = _e(je);
  {
    var dt = (ne) => {
      var Me = Fn(), pe = _e(Me);
      Va(pe, () => ce, !1, (he, J) => {
        sr(he, (He) => x(j, He), () => l(j)), ts(
          he,
          (He, ze) => ({
            "data-testid": _(),
            id: o(),
            class: `block ${He ?? ""}`,
            dir: O() ? "rtl" : "ltr",
            "aria-label": I(),
            style: "",
            [Dt]: {
              hidden: T() === "hidden",
              padded: h(),
              flex: y(),
              border_focus: f() === "focus",
              border_contrast: f() === "contrast",
              "hide-container": !w() && !m(),
              fullscreen: P(),
              animating: P() && l(X) !== null,
              "auto-margin": b() === null
            },
            [bt]: ze
          }),
          [
            () => (Oe(s()), oe(() => s()?.join(" ") || "")),
            () => ({
              height: (Oe(P()), Oe(r()), oe(() => P() ? void 0 : xe(r()))),
              "min-height": (Oe(P()), Oe(n()), oe(() => P() ? void 0 : xe(n()))),
              "max-height": (Oe(P()), Oe(i()), oe(() => P() ? void 0 : xe(i()))),
              "--start-top": (l(X), oe(() => l(X) ? `${l(X).top}px` : "0px")),
              "--start-left": (l(X), oe(() => l(X) ? `${l(X).left}px` : "0px")),
              "--start-width": (l(X), oe(() => l(X) ? `${l(X).width}px` : "0px")),
              "--start-height": (l(X), oe(() => l(X) ? `${l(X).height}px` : "0px")),
              width: (Oe(P()), Oe(a()), oe(() => P() ? void 0 : typeof a() == "number" ? `calc(min(${a()}px, 100%))` : xe(a()))),
              "border-style": c(),
              overflow: M() ? v() : "hidden",
              "flex-grow": b(),
              "min-width": `calc(min(${S()}px, 100%))`
            })
          ],
          void 0,
          void 0,
          "svelte-f3h6t9"
        );
        var ee = Fn(), tt = _e(ee);
        zr(tt, t, "default", {});
        var ie = Z(tt, 2);
        {
          var Ve = (He) => {
            var ze = Vo();
            vt("mousedown", ze, Ae), D(He, ze);
          };
          $(ie, (He) => {
            E() && He(Ve);
          });
        }
        D(J, ee);
      });
      var Le = Z(pe, 2);
      {
        var K = (he) => {
          var J = zo();
          let ee;
          re(() => ee = Ue(J, "", ee, {
            height: l(Ee) + "px",
            width: l(le) + "px"
          })), D(he, J);
        };
        $(Le, (he) => {
          P() && he(K);
        });
      }
      D(ne, Me);
    };
    $(ue, (ne) => {
      (T() === !0 || T() === "hidden") && ne(dt);
    });
  }
  D(e, je), cr();
}
var qo = /* @__PURE__ */ de('<span class="svelte-ajrv23"> </span>'), Wo = /* @__PURE__ */ de("<button><!> <div><!> <!></div></button>");
function jn(e, t) {
  let r = B(t, "label", 3, ""), n = B(t, "show_label", 3, !1), i = B(t, "pending", 3, !1), a = B(t, "size", 3, "small"), o = B(t, "padded", 3, !0), s = B(t, "highlight", 3, !1), c = B(t, "disabled", 3, !1), f = B(t, "hasPopup", 3, !1), h = B(t, "color", 3, "var(--block-label-text-color)"), p = B(t, "transparent", 3, !1), _ = B(t, "background", 3, "var(--block-background-fill)"), w = B(t, "border", 3, "transparent"), m = De(() => s() ? "var(--color-accent)" : h());
  var T = Wo();
  let M, v;
  var b = se(T);
  {
    var S = (U) => {
      var j = qo(), ce = se(j);
      re(() => be(ce, r())), D(U, j);
    };
    $(b, (U) => {
      n() && U(S);
    });
  }
  var y = Z(b, 2);
  let E;
  var O = se(y);
  Ga(O, () => t.Icon, (U, j) => {
    j(U, {});
  });
  var P = Z(O, 2);
  {
    var I = (U) => {
      var j = _t(), ce = _e(j);
      Na(ce, () => t.children), D(U, j);
    };
    $(P, (U) => {
      t.children && U(I);
    });
  }
  re(() => {
    M = et(T, 1, "icon-button svelte-ajrv23", null, M, {
      pending: i(),
      padded: o(),
      highlight: s(),
      transparent: p()
    }), T.disabled = c(), xt(T, "aria-label", r()), xt(T, "aria-haspopup", f()), xt(T, "title", r()), v = Ue(T, "", v, {
      "--border-color": w(),
      color: !c() && l(m) ? l(m) : "var(--block-label-text-color)",
      "--bg-color": c() ? "auto" : _()
    }), E = et(y, 1, "svelte-ajrv23", null, E, {
      "x-small": a() === "x-small",
      small: a() === "small",
      large: a() === "large",
      medium: a() === "medium"
    });
  }), Qe("click", T, function(...U) {
    t.onclick?.apply(this, U);
  }), D(e, T);
}
Ft(["click"]);
var Zo = /* @__PURE__ */ ii('<svg width="100%" height="100%" viewBox="0 0 24 24" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:space="preserve" stroke="currentColor" style="fill-rule:evenodd;clip-rule:evenodd;stroke-linecap:round;stroke-linejoin:round;"><g transform="matrix(1.14096,-0.140958,-0.140958,1.14096,-0.0559523,0.0559523)"><path d="M18,6L6.087,17.913" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></g><path d="M4.364,4.364L19.636,19.636" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></svg>');
function Vn(e) {
  var t = Zo();
  D(e, t);
}
Ft(["click"]);
function Dr(e) {
  let t = ["", "k", "M", "G", "T", "P", "E", "Z"], r = 0;
  for (; e > 1e3 && r < t.length - 1; )
    e /= 1e3, r++;
  let n = t[r];
  return (Number.isInteger(e) ? e : e.toFixed(1)) + n;
}
function zn(e) {
  return Object.prototype.toString.call(e) === "[object Date]";
}
function tn(e, t, r, n) {
  if (typeof r == "number" || zn(r)) {
    const i = n - r, a = (r - t) / (e.dt || 1 / 60), o = e.opts.stiffness * i, s = e.opts.damping * a, c = (o - s) * e.inv_mass, f = (a + c) * e.dt;
    return Math.abs(f) < e.opts.precision && Math.abs(i) < e.opts.precision ? n : (e.settled = !1, zn(r) ? new Date(r.getTime() + f) : r + f);
  } else {
    if (Array.isArray(r))
      return r.map(
        (i, a) => (
          // @ts-ignore
          tn(e, t[a], r[a], n[a])
        )
      );
    if (typeof r == "object") {
      const i = {};
      for (const a in r)
        i[a] = tn(e, t[a], r[a], n[a]);
      return i;
    } else
      throw new Error(`Cannot spring ${typeof r} values`);
  }
}
function Xn(e, t = {}) {
  const r = jt(e), { stiffness: n = 0.15, damping: i = 0.8, precision: a = 0.01 } = t;
  let o, s, c, f = (
    /** @type {T} */
    e
  ), h = (
    /** @type {T | undefined} */
    e
  ), p = 1, _ = 0, w = !1;
  function m(M, v = {}) {
    h = M;
    const b = c = {};
    return e == null || v.hard || T.stiffness >= 1 && T.damping >= 1 ? (w = !0, o = ke.now(), f = M, r.set(e = h), Promise.resolve()) : (v.soft && (_ = 1 / ((v.soft === !0 ? 0.5 : +v.soft) * 60), p = 0), s || (o = ke.now(), w = !1, s = ja((S) => {
      if (w)
        return w = !1, s = null, !1;
      p = Math.min(p + _, 1);
      const y = Math.min(S - o, 1e3 / 30), E = {
        inv_mass: p,
        opts: T,
        settled: !0,
        dt: y * 60 / 1e3
      }, O = tn(E, f, e, h);
      return o = S, f = /** @type {T} */
      e, r.set(e = /** @type {T} */
      O), E.settled && (s = null), !E.settled;
    })), new Promise((S) => {
      s.promise.then(() => {
        b === c && S();
      });
    }));
  }
  const T = {
    set: m,
    update: (M, v) => m(M(
      /** @type {T} */
      h,
      /** @type {T} */
      e
    ), v),
    subscribe: r.subscribe,
    stiffness: n,
    damping: i,
    precision: a
  };
  return T;
}
var Yo = /* @__PURE__ */ de('<div><svg viewBox="-1200 -1200 3000 3000" fill="none" xmlns="http://www.w3.org/2000/svg" class="svelte-1tt3t5l"><g><path d="M255.926 0.754768L509.702 139.936V221.027L255.926 81.8465V0.754768Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-1tt3t5l"></path><path d="M509.69 139.936L254.981 279.641V361.255L509.69 221.55V139.936Z" fill="#FF7C00" class="svelte-1tt3t5l"></path><path d="M0.250138 139.937L254.981 279.641V361.255L0.250138 221.55V139.937Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-1tt3t5l"></path><path d="M255.923 0.232622L0.236328 139.936V221.55L255.923 81.8469V0.232622Z" fill="#FF7C00" class="svelte-1tt3t5l"></path></g><g><path d="M255.926 141.5L509.702 280.681V361.773L255.926 222.592V141.5Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-1tt3t5l"></path><path d="M509.69 280.679L254.981 420.384V501.998L509.69 362.293V280.679Z" fill="#FF7C00" class="svelte-1tt3t5l"></path><path d="M0.250138 280.681L254.981 420.386V502L0.250138 362.295V280.681Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-1tt3t5l"></path><path d="M255.923 140.977L0.236328 280.68V362.294L255.923 222.591V140.977Z" fill="#FF7C00" class="svelte-1tt3t5l"></path></g></svg></div>');
function Jo(e, t) {
  hr(t, !0);
  const r = () => bn(c, "$top", i), n = () => bn(f, "$bottom", i), [i, a] = Pa();
  var o = this && this.__awaiter || function(S, y, E, O) {
    function P(I) {
      return I instanceof E ? I : new E(function(U) {
        U(I);
      });
    }
    return new (E || (E = Promise))(function(I, U) {
      function j(le) {
        try {
          Ee(O.next(le));
        } catch (X) {
          U(X);
        }
      }
      function ce(le) {
        try {
          Ee(O.throw(le));
        } catch (X) {
          U(X);
        }
      }
      function Ee(le) {
        le.done ? I(le.value) : P(le.value).then(j, ce);
      }
      Ee((O = O.apply(S, y || [])).next());
    });
  };
  let s = B(t, "margin", 3, !0);
  const c = Xn([0, 0]), f = Xn([0, 0]);
  let h = V(!1);
  function p() {
    return o(this, void 0, void 0, function* () {
      yield Promise.all([c.set([125, 140]), f.set([-125, -140])]), yield Promise.all([c.set([-125, 140]), f.set([125, -140])]), yield Promise.all([c.set([-125, 0]), f.set([125, -0])]), yield Promise.all([c.set([125, 0]), f.set([-125, 0])]);
    });
  }
  function _() {
    return o(this, void 0, void 0, function* () {
      yield p(), l(h) || _();
    });
  }
  function w() {
    return o(this, void 0, void 0, function* () {
      yield Promise.all([c.set([125, 0]), f.set([-125, 0])]), _();
    });
  }
  Be(() => (w(), () => {
    x(h, !0);
  }));
  var m = Yo();
  let T;
  var M = se(m), v = se(M), b = Z(v);
  re(() => {
    T = et(m, 1, "svelte-1tt3t5l", null, T, { margin: s() }), Ue(v, `transform: translate(${r()[0] ?? ""}px, ${r()[1] ?? ""}px);`), Ue(b, `transform: translate(${n()[0] ?? ""}px, ${n()[1] ?? ""}px);`);
  }), D(e, m), cr(), a();
}
var Qo = function(e, t, r, n) {
  function i(a) {
    return a instanceof r ? a : new r(function(o) {
      o(a);
    });
  }
  return new (r || (r = Promise))(function(a, o) {
    function s(h) {
      try {
        f(n.next(h));
      } catch (p) {
        o(p);
      }
    }
    function c(h) {
      try {
        f(n.throw(h));
      } catch (p) {
        o(p);
      }
    }
    function f(h) {
      h.done ? a(h.value) : i(h.value).then(s, c);
    }
    f((n = n.apply(e, t || [])).next());
  });
};
let er = [], kr = !1;
const Ko = typeof window < "u", Ni = Ko ? window.requestAnimationFrame : (e) => {
};
function $o(e) {
  return Qo(this, arguments, void 0, function* (t, r = !0) {
    if (!(window.__gradio_mode__ === "website" || window.__gradio_mode__ !== "app" && r !== !0)) {
      if (er.push(t), !kr) kr = !0;
      else return;
      yield wa(), Ni(() => {
        let n = [0, 0];
        for (let i = 0; i < er.length; i++) {
          const o = er[i].getBoundingClientRect();
          (i === 0 || o.top + window.scrollY <= n[0]) && (n[0] = o.top + window.scrollY, n[1] = i);
        }
        window.scrollTo({ top: n[0] - 20, behavior: "smooth" }), kr = !1, er = [];
      });
    }
  });
}
var el = /* @__PURE__ */ de('<div class="validation-error svelte-1qfla4m"> <button class="svelte-1qfla4m"><!></button></div>'), tl = /* @__PURE__ */ de('<div class="eta-bar svelte-1qfla4m"></div>'), rl = /* @__PURE__ */ de("<!> ", 1), nl = /* @__PURE__ */ de("<!> <!> <!> <!>", 1), il = /* @__PURE__ */ de('<div class="progress-level svelte-1qfla4m"><div class="progress-level-inner svelte-1qfla4m"><!></div> <div class="progress-bar-wrap svelte-1qfla4m"><div class="progress-bar svelte-1qfla4m"></div></div></div>'), al = /* @__PURE__ */ de('<p class="loading svelte-1qfla4m"> </p> <!>', 1), sl = /* @__PURE__ */ de("<!> <div><!> <!></div> <!> <!>", 1), ol = /* @__PURE__ */ de('<div class="clear-status svelte-1qfla4m"><!></div> <span class="error svelte-1qfla4m"> </span> <!>', 1), ll = /* @__PURE__ */ de("<div> <!> </div>"), ul = /* @__PURE__ */ de('<div data-testid="status-tracker"><!> <!></div> <!>', 1);
function fl(e, t) {
  hr(t, !0);
  let r = B(t, "eta", 3, null), n = B(t, "scroll_to_output", 3, !1), i = B(t, "timer", 3, !0), a = B(t, "show_progress", 3, "full"), o = B(t, "message", 3, null), s = B(t, "progress", 3, null), c = B(t, "variant", 3, "default"), f = B(t, "loading_text", 3, "Loading..."), h = B(t, "absolute", 3, !0), p = B(t, "translucent", 3, !1), _ = B(t, "border", 3, !1), w = B(t, "validation_error", 7, null), m = B(t, "show_validation_error", 3, !0), T = B(t, "type", 3, null), M = B(t, "used_cache", 3, null), v = B(t, "cache_duration", 3, null), b = B(t, "avg_time", 3, null), S, y = !1, E = V(0), O = V(null), P = V(null), I = V(!1), U = V(null), j = V(!1), ce = V(!1), Ee = V(null), le = V(null), X = V("from cache"), Se = V(!1), xe = null, Ae = null;
  const je = De(() => !(m() && w()) && (T() === "input" || !t.status || t.status === "complete" || a() === "hidden" || t.status == "streaming"));
  let ue = V(0);
  const dt = De(() => l(P) === null || l(P) <= 0 || !l(ue) ? 0 : Math.min(l(ue) / l(P), 1)), ne = De(() => l(ue).toFixed(1));
  let Me = De(() => s() == null), pe = De(() => r() !== null && r() !== void 0 ? r() : l(O));
  function Le() {
    Ni(() => {
      x(ue, (performance.now() - l(E)) / 1e3), y && Le();
    });
  }
  let K = De(() => {
    let z = null;
    s() != null ? z = s().map((te) => {
      if (te.index != null && te.length != null)
        return te.index / te.length;
      if (te.progress != null)
        return te.progress;
    }) : z = null;
    let Q, fe = "";
    return z ? (Q = z[z.length - 1], Q === 0 ? fe = "0" : fe = "150ms") : Q = void 0, {
      progress_level: z,
      last_progress_level: Q,
      progress_bar_transition: fe
    };
  });
  function he() {
    y || (x(O, x(U, null), !0), x(E, performance.now(), !0), y = !0, Le());
  }
  function J() {
    x(O, x(U, null), !0), y && (y = !1);
  }
  Be(() => {
    t.status === "pending" ? he() : oe(() => {
      J();
    });
  }), Be(() => {
    S && n() && (t.status === "pending" || t.status === "complete") && $o(S, t.autoscroll);
  }), Be(() => {
    l(pe) != null && l(O) !== l(pe) && (x(P, (performance.now() - l(E)) / 1e3 + l(pe)), x(U, l(P).toFixed(1), !0), x(O, l(pe), !0));
  });
  function ee() {
    x(I, !1);
  }
  Be(() => {
    oe(() => {
      ee();
    }), t.status === "error" && o() && x(I, !0);
  }), Be(() => {
    t.status === "complete" && T() === "output" && M() && v() != null && (x(Ee, v().toFixed(1), !0), x(X, M() === "full" ? "from cache" : "used cache", !0), x(Se, b() != null && b() > v() && b() > 0, !0), x(le, l(Se) ? b().toFixed(1) : null, !0), x(j, !0), x(ce, !1), xe && clearTimeout(xe), Ae && clearTimeout(Ae), xe = setTimeout(
      () => {
        x(ce, !0), Ae = setTimeout(
          () => {
            x(j, !1), x(ce, !1);
          },
          500
        );
      },
      1750
    ));
  });
  var tt = ul(), ie = _e(tt);
  let Ve, He;
  var ze = se(ie);
  {
    var rt = (z) => {
      var Q = el(), fe = se(Q), te = Z(fe), me = se(te);
      {
        let ge = De(() => t.i18n ? t.i18n("common.clear") : "Clear");
        jn(me, {
          get Icon() {
            return Vn;
          },
          get label() {
            return l(ge);
          },
          disabled: !1,
          size: "x-small",
          background: "var(--background-fill-primary)",
          color: "var(--error-background-text)",
          border: "var(--border-color-primary)",
          onclick: () => w(null)
        });
      }
      re(() => be(fe, `${w() ?? ""} `)), D(z, Q);
    };
    $(ze, (z) => {
      w() && m() && z(rt);
    });
  }
  var br = Z(ze, 2);
  {
    var It = (z) => {
      var Q = sl(), fe = _e(Q);
      {
        var te = (G) => {
          var q = tl();
          let we;
          re(() => we = Ue(q, "", we, {
            transform: `translateX(${(l(dt) || 0) * 100 - 100}%)`
          })), D(G, q);
        };
        $(fe, (G) => {
          c() === "default" && l(Me) && a() === "full" && G(te);
        });
      }
      var me = Z(fe, 2);
      let ge;
      var Ne = se(me);
      {
        var Ge = (G) => {
          var q = _t(), we = _e(q);
          xn(we, 17, s, _n, (it, Te) => {
            var pt = _t(), at = _e(pt);
            {
              var We = (Pe) => {
                var Ze = rl(), mt = _e(Ze);
                {
                  var Mt = (Re) => {
                    var Xe = Fe();
                    re((st, ot) => be(Xe, `${st ?? ""}/${ot ?? ""}`), [
                      () => Dr(l(Te).index || 0),
                      () => Dr(l(Te).length)
                    ]), D(Re, Xe);
                  }, ae = (Re) => {
                    var Xe = Fe();
                    re((st) => be(Xe, st), [() => Dr(l(Te).index || 0)]), D(Re, Xe);
                  };
                  $(mt, (Re) => {
                    l(Te).length != null ? Re(Mt) : Re(ae, -1);
                  });
                }
                var Ye = Z(mt);
                re(() => be(Ye, ` ${l(Te).unit ?? ""} |  `)), D(Pe, Ze);
              };
              $(at, (Pe) => {
                l(Te).index != null && Pe(We);
              });
            }
            D(it, pt);
          }), D(G, q);
        }, nt = (G) => {
          var q = Fe();
          re(() => be(q, `queue: ${t.queue_position + 1}/${t.queue_size ?? ""} |`)), D(G, q);
        }, Ce = (G) => {
          var q = Fe("processing |");
          D(G, q);
        };
        $(Ne, (G) => {
          s() ? G(Ge) : t.queue_position !== null && t.queue_size !== void 0 && t.queue_position >= 0 ? G(nt, 1) : t.queue_position === 0 && G(Ce, 2);
        });
      }
      var qt = Z(Ne, 2);
      {
        var Wt = (G) => {
          var q = Fe();
          re(() => be(q, `${l(ne) ?? ""}${r() ? `/${l(U)}` : ""}s`)), D(G, q);
        };
        $(qt, (G) => {
          i() && G(Wt);
        });
      }
      var Ot = Z(me, 2);
      {
        var Bt = (G) => {
          var q = il(), we = se(q), it = se(we);
          {
            var Te = (Pe) => {
              var Ze = _t(), mt = _e(Ze);
              xn(mt, 17, s, _n, (Mt, ae, Ye) => {
                var Re = _t(), Xe = _e(Re);
                {
                  var st = (ot) => {
                    var Zt = nl(), Yt = _e(Zt);
                    {
                      var wr = (g) => {
                        var H = Fe(" /");
                        D(g, H);
                      };
                      $(Yt, (g) => {
                        Ye !== 0 && g(wr);
                      });
                    }
                    var Jt = Z(Yt, 2);
                    {
                      var Tr = (g) => {
                        var H = Fe();
                        re(() => be(H, l(ae).desc)), D(g, H);
                      };
                      $(Jt, (g) => {
                        l(ae).desc != null && g(Tr);
                      });
                    }
                    var Qt = Z(Jt, 2);
                    {
                      var u = (g) => {
                        var H = Fe("-");
                        D(g, H);
                      };
                      $(Qt, (g) => {
                        l(ae).desc != null && l(K).progress_level && l(K).progress_level[Ye] != null && g(u);
                      });
                    }
                    var d = Z(Qt, 2);
                    {
                      var A = (g) => {
                        var H = Fe();
                        re((N) => be(H, `${N ?? ""}%`), [
                          () => (100 * (l(K).progress_level[Ye] || 0)).toFixed(1)
                        ]), D(g, H);
                      };
                      $(d, (g) => {
                        l(K).progress_level != null && g(A);
                      });
                    }
                    D(ot, Zt);
                  };
                  $(Xe, (ot) => {
                    (l(ae).desc != null || l(K).progress_level && l(K).progress_level[Ye] != null) && ot(st);
                  });
                }
                D(Mt, Re);
              }), D(Pe, Ze);
            };
            $(it, (Pe) => {
              s() != null && Pe(Te);
            });
          }
          var pt = Z(we, 2), at = se(pt);
          let We;
          re(() => We = Ue(at, "", We, {
            width: `${l(K).last_progress_level * 100}%`,
            transition: l(K).progress_bar_transition
          })), D(G, q);
        }, yr = (G) => {
          {
            let q = De(() => c() === "default");
            Jo(G, {
              get margin() {
                return l(q);
              }
            });
          }
        };
        $(Ot, (G) => {
          l(K).last_progress_level != null ? G(Bt) : a() === "full" && G(yr, 1);
        });
      }
      var xr = Z(Ot, 2);
      {
        var Er = (G) => {
          var q = al(), we = _e(q), it = se(we), Te = Z(we, 2);
          zr(Te, t, "additional-loading-text", {}), re(() => be(it, f())), D(G, q);
        };
        $(xr, (G) => {
          i() || G(Er);
        });
      }
      re(() => ge = et(me, 1, "progress-text svelte-1qfla4m", null, ge, {
        "meta-text-center": c() === "center",
        "meta-text": c() === "default"
      })), D(z, Q);
    }, zt = (z) => {
      var Q = ol(), fe = _e(Q), te = se(fe);
      {
        let Ge = De(() => t.i18n("common.clear"));
        jn(te, {
          get Icon() {
            return Vn;
          },
          get label() {
            return l(Ge);
          },
          disabled: !1,
          $$events: {
            click: () => {
              t.on_clear_status?.();
            }
          }
        });
      }
      var me = Z(fe, 2), ge = se(me), Ne = Z(me, 2);
      zr(Ne, t, "error", {}), re((Ge) => be(ge, Ge), [() => t.i18n("common.error")]), D(z, Q);
    };
    $(br, (z) => {
      t.status === "pending" ? z(It) : t.status === "error" && z(zt, 1);
    });
  }
  sr(ie, (z) => S = z, () => S);
  var Xt = Z(ie, 2);
  {
    var _r = (z) => {
      var Q = ll();
      let fe, te;
      var me = se(Q), ge = Z(me);
      {
        var Ne = (nt) => {
          var Ce = Fe();
          re(() => be(Ce, `~${l(le) ?? ""}s
			→ `)), D(nt, Ce);
        };
        $(ge, (nt) => {
          l(Se) && nt(Ne);
        });
      }
      var Ge = Z(ge);
      re(() => {
        fe = et(Q, 1, "cache-indicator svelte-1qfla4m", null, fe, { "fade-out": l(ce) }), te = Ue(Q, "", te, { position: h() ? "absolute" : "static" }), be(me, `⚡ ${l(X) ?? ""}: `), be(Ge, `${l(Ee) ?? ""}s`);
      }), D(z, Q);
    };
    $(Xt, (z) => {
      l(j) && z(_r);
    });
  }
  re(() => {
    Ve = et(ie, 1, `wrap ${c() ?? ""} ${a() ?? ""}`, "svelte-1qfla4m", Ve, {
      "no-click": w() && m(),
      hide: l(je),
      translucent: c() === "center" && (t.status === "pending" || t.status === "error") || p() || a() === "minimal" || w(),
      generating: t.status === "generating" && a() === "full",
      border: _()
    }), He = Ue(ie, "", He, {
      position: h() ? "absolute" : "static",
      padding: h() ? "0" : "var(--size-8) 0"
    });
  }), D(e, tt), cr();
}
const cl = (e) => {
  const t = {};
  for (let r = 0, n = e.length; r < n; r++) {
    const i = e[r];
    for (const a in i)
      t[a] ? t[a] = t[a].concat(i[a]) : t[a] = i[a];
  }
  return t;
}, hl = [
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
], dl = [
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
], pl = [
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
cl([
  Object.fromEntries(hl.map((e) => [e, ["*"]])),
  Object.fromEntries(dl.map((e) => [e, ["svg:*"]])),
  Object.fromEntries(pl.map((e) => [e, ["math:*"]]))
]);
Ft(["touchstart", "touchmove", "touchend", "click", "keydown"]);
var ml = /* @__PURE__ */ new Set(["$$slots", "$$events", "$$legacy"]), gl = /* @__PURE__ */ de('<!> <div class="stitch-preview svelte-1jifau7"><div class="canvas-wrap svelte-1jifau7"><canvas tabindex="0" role="application" aria-label="tile stitch preview canvas"></canvas></div> <div class="shortcut-bar svelte-1jifau7">↑↓←→ / WASD 步进 · Shift 10x · 普通拖动 1×原图倍率 · Shift+拖精准 0.25× · Space平移 · +/-缩放 · Esc取消 · Ctrl+Z撤销</div> <div class="status svelte-1jifau7"> </div></div>', 1);
function bl(e, t) {
  hr(t, !0);
  const r = /* @__PURE__ */ is(t, ml), n = 4, i = 70, a = 3, o = 30, s = 1e-3, c = 1, f = 0.25, h = new jo(r);
  let p, _, w = V(kt({ tiles: [], selected: 0 })), m = V(kt([])), T = V("点击画布以启用键盘"), M = V(!1), v = V("crosshair"), b = V(0), S = V(0), y = V(1), E = V(!1), O = V(!1), P = V(!1), I = V(-1), U = 0, j = 0, ce = 0, Ee = 0, le = 0, X = 0, Se = 0, xe = 0, Ae = !1, je = !1, ue = -1, dt = 0, ne = 0, Me = 0, pe = !1, Le = "", K = 0, he = null, J = [], ee = -1;
  function tt(u) {
    return typeof u == "number" ? String(u) + "px" : u || "520px";
  }
  function ie(u, d) {
    const A = Number(u);
    return Number.isFinite(A) ? A : d;
  }
  function Ve(u, d, A) {
    return Math.max(d, Math.min(A, u));
  }
  function He(u) {
    return JSON.parse(JSON.stringify(u || { tiles: [], selected: 0 }));
  }
  function ze(u) {
    var d;
    return {
      index: Math.trunc(ie(u.index, 0)),
      image: (d = u.image) !== null && d !== void 0 ? d : null,
      x: ie(u.x, 0),
      y: ie(u.y, 0),
      width: Math.max(1, ie(u.width, 1)),
      height: Math.max(1, ie(u.height, 1))
    };
  }
  function rt() {
    return Math.trunc(ie(l(w).selected, 0));
  }
  function br() {
    const u = ie(l(w).nudge_step, 1);
    return u > 0 ? u : 1;
  }
  function It() {
    const u = ie(l(w).drag_gain, c);
    return u > 0 ? u : c;
  }
  function zt() {
    return Math.min(It(), f);
  }
  function Xt() {
    return !!l(w).diff_mode;
  }
  function _r() {
    return l(w).show_loupe !== !1;
  }
  function z() {
    var u, d;
    const A = rt();
    for (const g of l(m))
      if (g.tile.index === A) return g.tile;
    return (d = (u = l(m)[0]) === null || u === void 0 ? void 0 : u.tile) !== null && d !== void 0 ? d : null;
  }
  function Q() {
    const u = z();
    if (!u) return { dx: 0, dy: 0 };
    const d = l(m).find((A) => A.tile.index === u.index);
    return d ? {
      dx: Math.round(u.x - d.baseX),
      dy: Math.round(u.y - d.baseY)
    } : { dx: 0, dy: 0 };
  }
  function fe() {
    return l(P) && je ? zt() : It();
  }
  function te() {
    const u = p?.getBoundingClientRect();
    return {
      width: Math.max(1, u?.width || 1),
      height: Math.max(1, u?.height || 1)
    };
  }
  function me(u, d) {
    return {
      x: (u - l(b)) / l(y),
      y: (d - l(S)) / l(y)
    };
  }
  function ge(u, d) {
    return {
      x: u * l(y) + l(b),
      y: d * l(y) + l(S)
    };
  }
  function Ne(u) {
    const d = p.getBoundingClientRect();
    return { x: u.clientX - d.left, y: u.clientY - d.top };
  }
  function Ge(u, d) {
    return d ? 90 / 255 : u === 0 ? 1 : 140 / 255;
  }
  function nt(u, d, A) {
    if (!u) {
      d === K && A(null);
      return;
    }
    const g = new Image();
    g.onload = () => {
      d === K && A(g);
    }, g.onerror = () => {
      d === K && A(null);
    }, g.src = u;
  }
  function Ce() {
    he && (clearTimeout(he), he = null);
  }
  function qt() {
    return {
      tiles: l(m).map((u) => ({
        index: u.tile.index,
        x: u.tile.x,
        y: u.tile.y
      })),
      selected: rt()
    };
  }
  function Wt(u, d) {
    return u.selected === d.selected && u.tiles.length === d.tiles.length && u.tiles.every((A, g) => {
      const H = d.tiles[g];
      return A.index === H.index && Math.abs(A.x - H.x) < 0.01 && Math.abs(A.y - H.y) < 0.01;
    });
  }
  function Ot() {
    const u = qt();
    ee >= 0 && Wt(J[ee], u) || (J = J.slice(0, ee + 1), J.push(u), J.length > o && J.shift(), ee = J.length - 1);
  }
  function Bt() {
    const u = qt();
    ee >= 0 && Wt(J[ee], u) || (J = J.slice(0, ee + 1), J.push(u), J.length > o && J.shift(), ee = J.length - 1);
  }
  function yr(u) {
    for (const d of u.tiles) {
      const A = l(m).find((g) => g.tile.index === d.index);
      A && (A.tile.x = d.x, A.tile.y = d.y);
    }
    x(w, Object.assign(Object.assign({}, l(w)), { selected: u.selected }), !0), x(m, [...l(m)], !0);
  }
  function xr() {
    if (!l(m).length) {
      x(b, 20), x(S, 20), x(y, 1);
      return;
    }
    let u = 1 / 0, d = 1 / 0, A = -1 / 0, g = -1 / 0;
    for (const Sr of l(m)) {
      const Ie = Sr.tile;
      u = Math.min(u, Ie.x), d = Math.min(d, Ie.y), A = Math.max(A, Ie.x + Ie.width), g = Math.max(g, Ie.y + Ie.height);
    }
    const H = 40, N = Math.max(1, A - u), L = Math.max(1, g - d), { width: C, height: W } = te(), ve = Ve(Math.min((C - H * 2) / N, (W - H * 2) / L), 0.05, 8);
    x(y, ve, !0), x(b, (C - (u + A) * ve) / 2), x(S, (W - (d + g) * ve) / 2);
  }
  function Er(u) {
    Ce(), K += 1;
    const d = K, A = new Map(l(m).map((L) => [L.tile.index, L])), g = He(u), H = Array.isArray(g.tiles) ? g.tiles.map((L) => {
      const C = ze(L);
      if (!C.image) {
        const W = A.get(C.index);
        W?.tile.image && (C.image = W.tile.image);
      }
      return C;
    }) : [];
    x(
      w,
      Object.assign(Object.assign({}, g), {
        tiles: H,
        selected: H.length ? Math.trunc(ie(g.selected, H[0].index)) : 0,
        nudge_step: ie(g.nudge_step, 1),
        diff_mode: !!g.diff_mode,
        show_loupe: g.show_loupe !== !1,
        drag_gain: ie(g.drag_gain, c),
        status: g.status || ""
      }),
      !0
    ), x(P, !1), x(I, -1), x(O, !1), J = [], ee = -1;
    const N = H.map((L) => ({
      tile: Object.assign({}, L),
      image: null,
      ready: !1,
      baseX: L.x,
      baseY: L.y
    }));
    x(m, N, !0), N.length && Bt(), x(T, l(w).status || (H.length ? "点击画布以启用键盘" : "等待 tile 数据"), !0);
    for (let L = 0; L < N.length; L++) {
      const C = N[L];
      nt(C.tile.image, d, (W) => {
        if (d !== K) return;
        const ve = l(m)[L];
        !ve || ve.tile.index !== C.tile.index || (ve.image = W, ve.ready = !!W, x(m, [...l(m)], !0), ae());
      });
    }
    requestAnimationFrame(() => xr());
  }
  Be(() => {
    const u = JSON.stringify(h.props.value || null);
    u !== Le && (Le = u, Er(h.props.value));
  }), Ca(() => {
    K += 1, Ce();
  }), ai(() => (window.addEventListener("blur", Ze), () => window.removeEventListener("blur", Ze)));
  function G(u) {
    var d;
    const A = Object.assign(Object.assign({}, l(w)), {
      tiles: l(m).map((g) => Object.assign({}, g.tile)),
      selected: rt(),
      status: (d = u ?? l(w).status) !== null && d !== void 0 ? d : ""
    });
    h.props.value = A, Le = JSON.stringify(A);
  }
  function q(u, d = !0) {
    x(w, Object.assign(Object.assign({}, l(w)), { status: u }), !0), x(T, u, !0), G(u), d && (Ce(), h.dispatch("change")), ae();
  }
  function we(u, d = 140) {
    q(u, !1), Ce(), he = setTimeout(
      () => {
        he = null, h.dispatch("change");
      },
      d
    );
  }
  function it(u, d) {
    for (let A = l(m).length - 1; A >= 0; A--) {
      const g = l(m)[A].tile;
      if (u >= g.x && u <= g.x + g.width && d >= g.y && d <= g.y + g.height)
        return g.index;
    }
    return -1;
  }
  function Te(u, d) {
    u < 0 || (x(w, Object.assign(Object.assign({}, l(w)), { selected: u }), !0), x(T, d || "已选择 tile " + String(u), !0), ae());
  }
  function pt(u, d) {
    const A = z();
    if (!A) return;
    Ot(), A.x += u, A.y += d, x(m, [...l(m)], !0), Bt();
    const g = Q();
    we("微调 tile " + String(A.index) + " → dx=" + String(g.dx) + " dy=" + String(g.dy));
  }
  function at(u, d, A) {
    const g = me(u, d);
    x(y, Ve(l(y) * A, 0.05, 16), !0);
    const H = ge(g.x, g.y);
    x(b, l(b) + (u - H.x)), x(S, l(S) + (d - H.y)), ae();
  }
  function We(u) {
    if (!(!p || u < 0))
      try {
        p.hasPointerCapture(u) && p.releasePointerCapture(u);
      } catch {
      }
  }
  function Pe(u = "已取消拖动") {
    const d = l(O) || ue >= 0 || l(I) >= 0, A = ue, g = l(m).find((H) => H.tile.index === l(I));
    g && (l(P) || Ae) && (g.tile.x = U, g.tile.y = j, x(m, [...l(m)], !0)), d && Ce(), x(w, Object.assign(Object.assign({}, l(w)), { selected: dt }), !0), x(P, !1), x(I, -1), Ae = !1, x(O, !1), ue = -1, je = !1, We(A), d && (x(T, u, !0), G(u), ae());
  }
  function Ze() {
    Pe("窗口失焦，已取消拖动");
  }
  function mt(u) {
    if (!_r() || !pe) return;
    const { width: d, height: A } = te(), g = Ve(ne, i + 2, d - i - 2), H = Ve(Me, i + 2, A - i - 2), N = me(g, H);
    u.save(), u.beginPath(), u.arc(g, H, i, 0, Math.PI * 2), u.clip(), u.fillStyle = "#0f172a", u.fillRect(g - i, H - i, i * 2, i * 2), u.translate(g, H), u.scale(a * l(y), a * l(y)), u.translate(-N.x, -N.y);
    for (const L of l(m)) {
      if (!L.ready || !L.image) continue;
      const C = L.tile, W = l(P) && l(I) === C.index;
      u.globalAlpha = Ge(C.index, W), Xt() && C.index !== 0 ? u.globalCompositeOperation = "difference" : u.globalCompositeOperation = "source-over", u.drawImage(L.image, C.x, C.y, C.width, C.height);
    }
    u.restore(), u.save(), u.beginPath(), u.arc(g, H, i, 0, Math.PI * 2), u.strokeStyle = "rgba(255,255,255,0.9)", u.lineWidth = 2, u.stroke(), u.strokeStyle = "rgba(15,23,42,0.85)", u.lineWidth = 1, u.beginPath(), u.moveTo(g - 8, H), u.lineTo(g + 8, H), u.moveTo(g, H - 8), u.lineTo(g, H + 8), u.stroke(), u.restore();
  }
  function Mt(u) {
    const d = z(), A = Q(), g = [
      "选中 #" + String(rt()),
      "dx " + String(A.dx) + "  dy " + String(A.dy),
      "zoom " + l(y).toFixed(2) + "  gain " + fe().toFixed(2)
    ];
    u.save(), u.font = "600 12px ui-monospace, SFMono-Regular, Menlo, monospace";
    const H = 8, N = 16, L = Math.max(...g.map((W) => u.measureText(W).width)) + H * 2, C = g.length * N + H;
    if (u.fillStyle = "rgba(15,23,42,0.82)", u.fillRect(10, 10, L, C), u.fillStyle = "#e2e8f0", g.forEach((W, ve) => {
      u.fillText(W, 10 + H, 10 + H + (ve + 1) * N - 4);
    }), d) {
      const W = ge(d.x, d.y), ve = ge(d.x + d.width, d.y + d.height);
      u.strokeStyle = "#22d3ee", u.lineWidth = 2, u.strokeRect(W.x, W.y, ve.x - W.x, ve.y - W.y);
    }
    u.restore();
  }
  function ae() {
    if (!p) return;
    const u = window.devicePixelRatio || 1, { width: d, height: A } = te();
    p.width = Math.max(1, Math.round(d * u)), p.height = Math.max(1, Math.round(A * u));
    const g = p.getContext("2d");
    if (g) {
      if (g.setTransform(u, 0, 0, u, 0, 0), g.clearRect(0, 0, d, A), g.fillStyle = "#0f172a", g.fillRect(0, 0, d, A), !l(m).length) {
        g.fillStyle = "#94a3b8", g.font = "16px sans-serif", g.fillText("等待 tile 数据", 24, 40);
        return;
      }
      g.save(), g.translate(l(b), l(S)), g.scale(l(y), l(y));
      for (const H of l(m)) {
        if (!H.ready || !H.image) continue;
        const N = H.tile, L = l(P) && l(I) === N.index;
        g.globalAlpha = Ge(N.index, L), Xt() && N.index !== 0 ? g.globalCompositeOperation = "difference" : g.globalCompositeOperation = "source-over", g.drawImage(H.image, N.x, N.y, N.width, N.height);
      }
      if (g.restore(), l(P) && l(I) >= 0) {
        const H = l(m).find((N) => N.tile.index === l(I));
        if (H) {
          const N = ge(ce, Ee), L = ge(ce + H.tile.width, Ee + H.tile.height);
          g.save(), g.setLineDash([6, 4]), g.strokeStyle = "rgba(250,204,21,0.95)", g.lineWidth = 2, g.strokeRect(N.x, N.y, L.x - N.x, L.y - N.y), g.restore();
        }
      }
      Mt(g), mt(g);
    }
  }
  function Ye() {
    x(M, !0), x(T, "键盘已接管");
  }
  function Re() {
    Pe(), x(M, !1), x(E, !1), !l(P) && !l(O) && x(v, "crosshair"), x(T, l(w).status || "点击画布以启用键盘", !0);
  }
  function Xe() {
    p.focus();
  }
  function st(u) {
    if (!l(M)) return;
    const d = u.key, A = d.toLowerCase(), g = /* @__PURE__ */ new Set([
      "ArrowUp",
      "ArrowDown",
      "ArrowLeft",
      "ArrowRight",
      "w",
      "a",
      "s",
      "d",
      " ",
      "+",
      "=",
      "-",
      "_",
      "Escape",
      "z"
    ]);
    if ((g.has(d) || g.has(A) || u.ctrlKey && A === "z") && u.preventDefault(), d === " " || d === "Spacebar") {
      x(E, !0), x(v, "grab");
      return;
    }
    if (d === "Escape") {
      (l(P) || l(O) || ue >= 0) && Pe("已取消拖动");
      return;
    }
    if (u.ctrlKey && A === "z") {
      ee > 0 && (ee -= 1, yr(J[ee]), we("撤销到步骤 " + String(ee + 1)));
      return;
    }
    if (d === "+" || d === "=") {
      const L = te();
      at(L.width / 2, L.height / 2, 1.15);
      return;
    }
    if (d === "-" || d === "_") {
      const L = te();
      at(L.width / 2, L.height / 2, 1 / 1.15);
      return;
    }
    let H = 0, N = 0;
    if ((d === "ArrowLeft" || A === "a") && (H = -1), (d === "ArrowRight" || A === "d") && (H = 1), (d === "ArrowUp" || A === "w") && (N = -1), (d === "ArrowDown" || A === "s") && (N = 1), H !== 0 || N !== 0) {
      const L = br() * (u.shiftKey ? 10 : 1);
      pt(H * L, N * L);
    }
  }
  function ot(u) {
    l(M) && (u.key === " " || u.key === "Spacebar") && (x(E, !1), l(O) || x(v, l(P) ? "grabbing" : "crosshair", !0));
  }
  function Zt(u) {
    if (!p) return;
    Ce();
    const d = Ne(u);
    if (le = d.x, X = d.y, Se = d.x, xe = d.y, ne = d.x, Me = d.y, pe = !0, je = u.shiftKey, dt = rt(), u.button === 1 || u.button === 0 && l(E)) {
      x(O, !0), ue = u.pointerId, x(v, "grabbing"), p.setPointerCapture(u.pointerId);
      return;
    }
    if (u.button !== 0) return;
    const A = me(d.x, d.y), g = it(A.x, A.y);
    if (g >= 0) {
      Te(g);
      const H = l(m).find((N) => N.tile.index === g);
      H && (x(I, g, !0), U = H.tile.x, j = H.tile.y, ce = H.tile.x, Ee = H.tile.y, x(P, !1), Ae = !1, ue = u.pointerId, p.setPointerCapture(u.pointerId));
    } else
      x(I, -1);
    ae();
  }
  function Yt(u) {
    const d = Ne(u);
    if (ne = d.x, Me = d.y, pe = !0, l(O)) {
      x(b, l(b) + (d.x - Se)), x(S, l(S) + (d.y - xe)), Se = d.x, xe = d.y, x(v, "grabbing"), ae();
      return;
    }
    if (ue === u.pointerId && l(I) >= 0) {
      const A = Math.hypot(d.x - le, d.y - X), g = l(m).find((H) => H.tile.index === l(I));
      if (!g) return;
      if (!l(P) && A >= n && (x(P, !0), Ot(), x(v, "grabbing")), l(P)) {
        je = u.shiftKey;
        const H = u.shiftKey ? zt() : It(), N = (d.x - Se) / l(y) * H, L = (d.y - xe) / l(y) * H;
        g.tile.x += N, g.tile.y += L, x(m, [...l(m)], !0), Ae = !0;
        const C = Q();
        x(T, "拖动 tile " + String(l(I)) + "  dx=" + String(C.dx) + " dy=" + String(C.dy)), x(w, Object.assign(Object.assign({}, l(w)), { status: l(T) }), !0), G(l(T));
      }
      Se = d.x, xe = d.y, ae();
      return;
    }
    l(E) ? x(v, "grab") : x(v, "crosshair"), ae();
  }
  function wr(u) {
    if (l(O)) {
      x(O, !1), We(u.pointerId), x(v, l(E) ? "grab" : "crosshair", !0), ue = -1, ae();
      return;
    }
    if (ue === u.pointerId && l(I) >= 0) {
      const d = Ne(u), A = Math.hypot(d.x - le, d.y - X);
      if (!l(P) && A < n)
        Te(l(I), "已选择 tile " + String(l(I))), q("已选择 tile " + String(l(I)));
      else if (l(P) && Ae) {
        Bt();
        const g = Q();
        q("tile " + String(l(I)) + " 对齐 dx=" + String(g.dx) + " dy=" + String(g.dy));
      }
      x(P, !1), x(I, -1), Ae = !1, ue = -1, We(u.pointerId), x(v, l(E) ? "grab" : "crosshair", !0), ae();
    }
  }
  function Jt() {
    Pe();
  }
  function Tr() {
    pe = !1, !l(O) && !l(P) && x(v, l(E) ? "grab" : "crosshair", !0), ae();
  }
  function Qt(u) {
    u.preventDefault();
    const d = Ne(u), A = Math.exp(-u.deltaY * (u.ctrlKey ? s * 4 : s));
    at(d.x, d.y, A);
  }
  {
    let u = De(() => l(M) ? "focus" : "base");
    Xo(e, {
      get visible() {
        return h.shared.visible;
      },
      variant: "solid",
      get border_mode() {
        return l(u);
      },
      padding: !1,
      get elem_id() {
        return h.shared.elem_id;
      },
      get elem_classes() {
        return h.shared.elem_classes;
      },
      allow_overflow: !1,
      get container() {
        return h.shared.container;
      },
      get scale() {
        return h.shared.scale;
      },
      get min_width() {
        return h.shared.min_width;
      },
      children: (d, A) => {
        var g = gl(), H = _e(g);
        fl(H, ss(
          {
            get autoscroll() {
              return h.shared.autoscroll;
            },
            get i18n() {
              return h.i18n;
            }
          },
          () => h.shared.loading_status,
          {
            on_clear_status: () => h.dispatch("clear_status", h.shared.loading_status)
          }
        ));
        var N = Z(H, 2), L = se(N), C = se(L);
        let W;
        sr(C, (Ie) => p = Ie, () => p), sr(L, (Ie) => _ = Ie, () => _);
        var ve = Z(L, 4), Sr = se(ve);
        re(
          (Ie) => {
            Ue(N, Ie), Ue(C, "cursor:" + l(v)), W = et(C, 1, "svelte-1jifau7", null, W, { focused: l(M) }), be(Sr, l(T));
          },
          [() => "height:" + tt(h.props.height)]
        ), vt("focus", C, Ye), vt("blur", C, Re), Qe("click", C, Xe), Qe("keydown", C, st), Qe("keyup", C, ot), Qe("pointerdown", C, Zt), Qe("pointermove", C, Yt), Qe("pointerup", C, wr), vt("pointercancel", C, Jt), vt("pointerleave", C, Tr), vt("wheel", C, Qt), D(d, g);
      },
      $$slots: { default: !0 }
    });
  }
  cr();
}
Ft([
  "click",
  "keydown",
  "keyup",
  "pointerdown",
  "pointermove",
  "pointerup"
]);
export {
  bl as default
};

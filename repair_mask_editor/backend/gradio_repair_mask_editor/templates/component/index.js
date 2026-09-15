import { i as Gr, g as Ln, o as Bi, n as We, u as ne, s as Pi, r as wr, m as Qe, a as D, b as g, t as jr, d as Mi, q as Oi, c as Nn, e as et, f as qt, h as jt, j as Ci, T as Li, k as Ni, l as Vt, p as Ke, v as Vr, w as tt, x as Rn, y as Dn, z as kn, A as At, E as Yt, B as ot, C as Un, D as Ee, F as en, G as Ri, H as Fn, I as zr, J as Di, K as tn, L as ki, M as Ui, N as ke, O as Gn, P as fr, Q as Fi, R as Gi, S as ji, U as Vi, V as Jt, W as jn, X as rn, Y as nn, Z as zi, _ as Xi, $ as Wi, a0 as Zi, a1 as qi, a2 as Yi, a3 as Ji, a4 as Qi, a5 as Xr, a6 as Ki, a7 as Tt, a8 as Ht, a9 as $i, aa as ea, ab as ta, ac as ra, ad as na, ae as ia, af as Wr, ag as aa, ah as sa, ai as xe, aj as Tr, ak as Sr, al as oa, am as la, an as zt, ao as ua, ap as fa, aq as ha, ar as ca, as as da, at as Vn, au as _t, av as pa, aw as an, ax as va, ay as pe, az as Qt, aA as Kt, aB as Z, aC as Ar, aD as K, aE as ma, aF as se, aG as ge, aH as Oe, aI as le, aJ as ga } from "./render-PkCSbJiD.js";
function zn(e) {
  throw new Error("https://svelte.dev/e/lifecycle_outside_component");
}
const ba = [];
function _a(e, t = !1, r = !1) {
  return Ut(e, /* @__PURE__ */ new Map(), "", ba, null, r);
}
function Ut(e, t, r, n, i = null, a = !1) {
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
    if (Gr(e)) {
      var s = (
        /** @type {Snapshot<any>} */
        Array(e.length)
      );
      t.set(e, s), i !== null && t.set(i, s);
      for (var u = 0; u < e.length; u += 1) {
        var l = e[u];
        u in e && (s[u] = Ut(l, t, r, n, null, a));
      }
      return s;
    }
    if (Ln(e) === Bi) {
      s = {}, t.set(e, s), i !== null && t.set(i, s);
      for (var h of Object.keys(e))
        s[h] = Ut(
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
      return Ut(
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
function Zr(e, t, r) {
  if (e == null)
    return t(void 0), r && r(void 0), We;
  const n = ne(
    () => e.subscribe(
      t,
      // @ts-expect-error
      r
    )
  );
  return n.unsubscribe ? () => n.unsubscribe() : n;
}
const it = [];
function ya(e, t) {
  return {
    subscribe: It(e, t).subscribe
  };
}
function It(e, t = We) {
  let r = null;
  const n = /* @__PURE__ */ new Set();
  function i(s) {
    if (Pi(e, s) && (e = s, r)) {
      const u = !it.length;
      for (const l of n)
        l[1](), it.push(l, e);
      if (u) {
        for (let l = 0; l < it.length; l += 2)
          it[l][0](it[l + 1]);
        it.length = 0;
      }
    }
  }
  function a(s) {
    i(s(
      /** @type {T} */
      e
    ));
  }
  function o(s, u = We) {
    const l = [s, u];
    return n.add(l), n.size === 1 && (r = t(i, a) || We), s(
      /** @type {T} */
      e
    ), () => {
      n.delete(l), n.size === 0 && r && (r(), r = null);
    };
  }
  return { set: i, update: a, subscribe: o };
}
function dt(e, t, r) {
  const n = !Array.isArray(e), i = n ? [e] : e;
  if (!i.every(Boolean))
    throw new Error("derived() expects stores as input, got a falsy value");
  const a = t.length < 2;
  return ya(r, (o, s) => {
    let u = !1;
    const l = [];
    let h = 0, c = We;
    const m = () => {
      if (h)
        return;
      c();
      const d = t(n ? l[0] : l, o, s);
      a ? o(d) : c = typeof d == "function" ? d : We;
    }, T = i.map(
      (d, _) => Zr(
        d,
        (P) => {
          l[_] = P, h &= ~(1 << _), u && m();
        },
        () => {
          h |= 1 << _;
        }
      )
    );
    return u = !0, m(), function() {
      wr(T), c(), u = !1;
    };
  });
}
function xa(e) {
  let t;
  return Zr(e, (r) => t = r)(), t;
}
let Rt = !1, Hr = /* @__PURE__ */ Symbol("unmounted");
function sn(e, t, r) {
  const n = r[t] ??= {
    store: null,
    source: Qe(void 0),
    unsubscribe: We
  };
  if (n.store !== e && !(Hr in r))
    if (n.unsubscribe(), n.store = e ?? null, e == null)
      n.source.v = void 0, n.unsubscribe = We;
    else {
      var i = !0;
      n.unsubscribe = Zr(e, (a) => {
        i ? n.source.v = a : D(n.source, a);
      }), i = !1;
    }
  return e && Hr in r ? xa(e) : g(n.source);
}
function Ea() {
  const e = {};
  function t() {
    jr(() => {
      for (var r in e)
        e[r].unsubscribe();
      Mi(e, Hr, {
        enumerable: !1,
        value: !0
      });
    });
  }
  return [e, t];
}
function wa(e) {
  var t = Rt;
  try {
    return Rt = !1, [e(), Rt];
  } finally {
    Rt = t;
  }
}
function Ta(e, t) {
  if (t) {
    const r = document.body;
    e.autofocus = !0, Oi(() => {
      document.activeElement === r && e.focus();
    });
  }
}
const Sa = (
  // We gotta write it like this because after downleveling the pure comment may end up in the wrong location
  globalThis?.window?.trustedTypes && /* @__PURE__ */ globalThis.window.trustedTypes.createPolicy("svelte-trusted-html", {
    /** @param {string} html */
    createHTML: (e) => e
  })
);
function Aa(e) {
  return (
    /** @type {string} */
    Sa?.createHTML(e) ?? e
  );
}
function Xn(e) {
  var t = Nn("template");
  return t.innerHTML = Aa(e.replaceAll("<!>", "<!---->")), t.content;
}
function ut(e, t) {
  var r = (
    /** @type {Effect} */
    qt
  );
  r.nodes === null && (r.nodes = { start: e, end: t, a: null, t: null });
}
// @__NO_SIDE_EFFECTS__
function fe(e, t) {
  var r = (t & Li) !== 0, n = (t & Ni) !== 0, i, a = !e.startsWith("<!>");
  return () => {
    i === void 0 && (i = Xn(a ? e : "<!>" + e), r || (i = /** @type {TemplateNode} */
    jt(i)));
    var o = (
      /** @type {TemplateNode} */
      n || Ci ? document.importNode(i, !0) : i.cloneNode(!0)
    );
    if (r) {
      var s = (
        /** @type {TemplateNode} */
        jt(o)
      ), u = (
        /** @type {TemplateNode} */
        o.lastChild
      );
      ut(s, u);
    } else
      ut(o, o);
    return o;
  };
}
// @__NO_SIDE_EFFECTS__
function Ha(e, t, r = "svg") {
  var n = !e.startsWith("<!>"), i = `<${r}>${n ? e : "<!>" + e}</${r}>`, a;
  return () => {
    if (!a) {
      var o = (
        /** @type {DocumentFragment} */
        Xn(i)
      ), s = (
        /** @type {Element} */
        jt(o)
      );
      a = /** @type {Element} */
      jt(s);
    }
    var u = (
      /** @type {TemplateNode} */
      a.cloneNode(!0)
    );
    return ut(u, u), u;
  };
}
// @__NO_SIDE_EFFECTS__
function Wn(e, t) {
  return /* @__PURE__ */ Ha(e, t, "svg");
}
function Me(e = "") {
  {
    var t = et(e + "");
    return ut(t, t), t;
  }
}
function st() {
  var e = document.createDocumentFragment(), t = document.createComment(""), r = et();
  return e.append(t, r), ut(t, r), e;
}
function N(e, t) {
  e !== null && e.before(
    /** @type {Node} */
    t
  );
}
class $t {
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
        Vt(n), this.#n.delete(r);
      else {
        var i = this.#e.get(r);
        i && (Vt(i.effect), this.#r.set(r, i.effect), this.#e.delete(r), i.fragment.lastChild.remove(), this.anchor.before(i.fragment), n = i.effect);
      }
      for (const [a, o] of this.#t) {
        if (this.#t.delete(a), a === t)
          break;
        const s = this.#e.get(o);
        s && (Ke(s.effect), this.#e.delete(o));
      }
      for (const [a, o] of this.#r) {
        if (a === r || this.#n.has(a)) continue;
        const s = () => {
          if (Array.from(this.#t.values()).includes(a)) {
            var l = document.createDocumentFragment();
            Dn(o, l), l.append(et()), this.#e.set(a, { effect: o, fragment: l });
          } else
            Ke(o);
          this.#n.delete(a), this.#r.delete(a);
        };
        this.#i || !n ? (this.#n.add(a), Vr(o, s, !1)) : s();
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
      r.includes(n) || (Ke(i.effect), this.#e.delete(n));
  };
  /**
   *
   * @param {any} key
   * @param {null | ((target: TemplateNode) => void)} fn
   */
  ensure(t, r) {
    var n = (
      /** @type {Batch} */
      Rn
    ), i = kn();
    if (r && !this.#r.has(t) && !this.#e.has(t))
      if (i) {
        var a = document.createDocumentFragment(), o = et();
        a.append(o), this.#e.set(t, {
          effect: tt(() => r(o)),
          fragment: a
        });
      } else
        this.#r.set(
          t,
          tt(() => r(this.anchor))
        );
    if (this.#t.set(n, t), i) {
      for (const [s, u] of this.#r)
        s === t ? n.unskip_effect(u) : n.skip_effect(u);
      for (const [s, u] of this.#e)
        s === t ? n.unskip_effect(u.effect) : n.skip_effect(u.effect);
      n.oncommit(this.#a), n.ondiscard(this.#s);
    } else
      this.#a(n);
  }
}
function Ia(e, t, ...r) {
  var n = new $t(e);
  At(() => {
    const i = t() ?? null;
    n.ensure(i, i && ((a) => i(a, ...r)));
  }, Yt);
}
function Zn(e) {
  ot === null && zn(), Un && ot.l !== null ? Pa(ot).m.push(e) : Ee(() => {
    const t = ne(e);
    if (typeof t == "function") return (
      /** @type {() => void} */
      t
    );
  });
}
function Ba(e) {
  ot === null && zn(), Zn(() => () => ne(e));
}
function Pa(e) {
  var t = (
    /** @type {ComponentContextLegacy} */
    e.l
  );
  return t.u ??= { a: [], b: [], m: [] };
}
function Y(e, t, r = !1) {
  var n = new $t(e), i = r ? Yt : 0;
  function a(o, s) {
    n.ensure(o, s);
  }
  At(() => {
    var o = !1;
    t((s, u = 0) => {
      o = !0, a(u, s);
    }), o || a(-1, null);
  }, i);
}
function on(e, t) {
  return t;
}
function Ma(e, t, r) {
  for (var n = [], i = t.length, a, o = t.length, s = 0; s < i; s++) {
    let c = t[s];
    Vr(
      c,
      () => {
        if (a) {
          if (a.pending.delete(c), a.done.add(c), a.pending.size === 0) {
            var m = (
              /** @type {Set<EachOutroGroup>} */
              e.outrogroups
            );
            Ir(e, zr(a.done)), m.delete(a), m.size === 0 && (e.outrogroups = null);
          }
        } else
          o -= 1;
      },
      !1
    );
  }
  if (o === 0) {
    var u = n.length === 0 && r !== null;
    if (u) {
      var l = (
        /** @type {Element} */
        r
      ), h = (
        /** @type {Element} */
        l.parentNode
      );
      Gi(h), h.append(l), e.items.clear();
    }
    Ir(e, t, !u);
  } else
    a = {
      pending: new Set(t),
      done: /* @__PURE__ */ new Set()
    }, (e.outrogroups ??= /* @__PURE__ */ new Set()).add(a);
}
function Ir(e, t, r = !0) {
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
      a.f |= ke;
      const o = document.createDocumentFragment();
      Dn(a, o);
    } else
      Ke(t[i], r);
  }
}
var ln;
function un(e, t, r, n, i, a = null) {
  var o = e, s = /* @__PURE__ */ new Map(), u = null, l = Fn(() => {
    var p = r();
    return (
      /** @type {V[]} */
      Gr(p) ? p : p == null ? [] : zr(p)
    );
  }), h, c = /* @__PURE__ */ new Map(), m = !0;
  function T(p) {
    (P.effect.f & Gn) === 0 && (P.pending.delete(p), P.fallback = u, Oa(P, h, o, t, n), u !== null && (h.length === 0 ? (u.f & ke) === 0 ? Vt(u) : (u.f ^= ke, Et(u, null, o)) : Vr(u, () => {
      u = null;
    })));
  }
  function d(p) {
    P.pending.delete(p);
  }
  var _ = At(() => {
    h = /** @type {V[]} */
    g(l);
    for (var p = h.length, b = /* @__PURE__ */ new Set(), w = (
      /** @type {Batch} */
      Rn
    ), E = kn(), x = 0; x < p; x += 1) {
      var A = h[x], H = n(A, x), B = m ? null : s.get(H);
      B ? (B.v && en(B.v, A), B.i && en(B.i, x), E && w.unskip_effect(B.e)) : (B = Ca(
        s,
        m ? o : ln ??= et(),
        A,
        H,
        x,
        i,
        t,
        r
      ), m || (B.e.f |= ke), s.set(H, B)), b.add(H);
    }
    if (p === 0 && a && !u && (m ? u = tt(() => a(o)) : (u = tt(() => a(ln ??= et())), u.f |= ke)), p > b.size && Ri(), !m)
      if (c.set(w, b), E) {
        for (const [O, U] of s)
          b.has(O) || w.skip_effect(U.e);
        w.oncommit(T), w.ondiscard(d);
      } else
        T(w);
    g(l);
  }), P = { effect: _, items: s, pending: c, outrogroups: null, fallback: u };
  m = !1;
}
function yt(e) {
  for (; e !== null && (e.f & Fi) === 0; )
    e = e.next;
  return e;
}
function Oa(e, t, r, n, i) {
  var a = t.length, o = e.items, s = yt(e.effect.first), u, l = null, h = [], c = [], m, T, d, _;
  for (_ = 0; _ < a; _ += 1) {
    if (m = t[_], T = i(m, _), d = /** @type {EachItem} */
    o.get(T).e, e.outrogroups !== null)
      for (const B of e.outrogroups)
        B.pending.delete(d), B.done.delete(d);
    if ((d.f & fr) !== 0 && Vt(d), (d.f & ke) !== 0)
      if (d.f ^= ke, d === s)
        Et(d, null, r);
      else {
        var P = l ? l.next : s;
        d === e.effect.last && (e.effect.last = d.prev), d.prev && (d.prev.next = d.next), d.next && (d.next.prev = d.prev), ze(e, l, d), ze(e, d, P), Et(d, P, r), l = d, h = [], c = [], s = yt(l.next);
        continue;
      }
    if (d !== s) {
      if (u !== void 0 && u.has(d)) {
        if (h.length < c.length) {
          var p = c[0], b;
          l = p.prev;
          var w = h[0], E = h[h.length - 1];
          for (b = 0; b < h.length; b += 1)
            Et(h[b], p, r);
          for (b = 0; b < c.length; b += 1)
            u.delete(c[b]);
          ze(e, w.prev, E.next), ze(e, l, w), ze(e, E, p), s = p, l = E, _ -= 1, h = [], c = [];
        } else
          u.delete(d), Et(d, s, r), ze(e, d.prev, d.next), ze(e, d, l === null ? e.effect.first : l.next), ze(e, l, d), l = d;
        continue;
      }
      for (h = [], c = []; s !== null && s !== d; )
        (u ??= /* @__PURE__ */ new Set()).add(s), c.push(s), s = yt(s.next);
      if (s === null)
        continue;
    }
    (d.f & ke) === 0 && h.push(d), l = d, s = yt(d.next);
  }
  if (e.outrogroups !== null) {
    for (const B of e.outrogroups)
      B.pending.size === 0 && (Ir(e, zr(B.done)), e.outrogroups?.delete(B));
    e.outrogroups.size === 0 && (e.outrogroups = null);
  }
  if (s !== null || u !== void 0) {
    var x = [];
    if (u !== void 0)
      for (d of u)
        (d.f & fr) === 0 && x.push(d);
    for (; s !== null; )
      (s.f & fr) === 0 && s !== e.fallback && x.push(s), s = yt(s.next);
    var A = x.length;
    if (A > 0) {
      var H = null;
      Ma(e, x, H);
    }
  }
}
function Ca(e, t, r, n, i, a, o, s) {
  var u = (o & ki) !== 0 ? (o & Ui) === 0 ? Qe(r, !1, !1) : tn(r) : null, l = (o & Di) !== 0 ? tn(i) : null;
  return {
    v: u,
    i: l,
    e: tt(() => (a(t, u ?? r, l ?? i, s), () => {
      e.delete(n);
    }))
  };
}
function Et(e, t, r) {
  if (e.nodes)
    for (var n = e.nodes.start, i = e.nodes.end, a = t && (t.f & ke) === 0 ? (
      /** @type {EffectNodes} */
      t.nodes.start
    ) : r; n !== null; ) {
      var o = (
        /** @type {TemplateNode} */
        ji(n)
      );
      if (a.before(n), n === i)
        return;
      n = o;
    }
}
function ze(e, t, r) {
  t === null ? e.effect.first = r : t.next = r, r === null ? e.effect.last = t : r.prev = t;
}
function Br(e, t, r, n, i) {
  var a = t.$$slots?.[r], o = !1;
  a === !0 && (a = t[r === "default" ? "children" : r], o = !0), a === void 0 || a(e, o ? () => n : n);
}
function La(e, t, r) {
  var n = new $t(e);
  At(() => {
    var i = t() ?? null;
    n.ensure(i, i && ((a) => r(a, i)));
  }, Yt);
}
const Na = () => performance.now(), Te = {
  // don't access requestAnimationFrame eagerly outside method
  // this allows basic testing of user code without JSDOM
  // bunder will eval and remove ternary when the user's app is built
  tick: (
    /** @param {any} _ */
    (e) => requestAnimationFrame(e)
  ),
  now: () => Na(),
  tasks: /* @__PURE__ */ new Set()
};
function qn() {
  const e = Te.now();
  Te.tasks.forEach((t) => {
    t.c(e) || (Te.tasks.delete(t), t.f());
  }), Te.tasks.size !== 0 && Te.tick(qn);
}
function Ra(e) {
  let t;
  return Te.tasks.size === 0 && Te.tick(qn), {
    promise: new Promise((r) => {
      Te.tasks.add(t = { c: e, f: r });
    }),
    abort() {
      Te.tasks.delete(t);
    }
  };
}
function Da(e, t, r, n, i, a) {
  var o = null, s = (
    /** @type {TemplateNode} */
    e
  ), u = new $t(s, !1);
  At(() => {
    const l = t() || null;
    var h = l === "svg" ? Vi : void 0;
    if (l === null) {
      u.ensure(null, null);
      return;
    }
    return u.ensure(l, (c) => {
      if (l) {
        if (o = Nn(l, h), ut(o, o), n) {
          var m = null, T = o.appendChild(et());
          n(o, T), m?.remove();
        }
        qt.nodes.end = o, c.before(o);
      }
    }), () => {
    };
  }, Yt), jr(() => {
  });
}
function ka(e, t, r) {
  Jt(() => {
    var n = ne(() => t(e, r?.()) || {});
    if (n?.destroy)
      return () => (
        /** @type {Function} */
        n.destroy()
      );
  });
}
function Ua(e, t) {
  var r = void 0, n;
  jn(() => {
    r !== (r = t()) && (n && (Ke(n), n = null), r && (n = tt(() => {
      Jt(() => (
        /** @type {(node: Element) => void} */
        r(e)
      ));
    })));
  });
}
function Yn(e) {
  var t, r, n = "";
  if (typeof e == "string" || typeof e == "number") n += e;
  else if (typeof e == "object") if (Array.isArray(e)) {
    var i = e.length;
    for (t = 0; t < i; t++) e[t] && (r = Yn(e[t])) && (n && (n += " "), n += r);
  } else for (r in e) e[r] && (n && (n += " "), n += r);
  return n;
}
function Fa() {
  for (var e, t, r = 0, n = "", i = arguments.length; r < i; r++) (e = arguments[r]) && (t = Yn(e)) && (n && (n += " "), n += t);
  return n;
}
function Ga(e) {
  return typeof e == "object" ? Fa(e) : e ?? "";
}
const fn = [...`
\r\f \v\uFEFF`];
function ja(e, t, r) {
  var n = e == null ? "" : "" + e;
  if (t && (n = n ? n + " " + t : t), r) {
    for (var i of Object.keys(r))
      if (r[i])
        n = n ? n + " " + i : i;
      else if (n.length)
        for (var a = i.length, o = 0; (o = n.indexOf(i, o)) >= 0; ) {
          var s = o + a;
          (o === 0 || fn.includes(n[o - 1])) && (s === n.length || fn.includes(n[s])) ? n = (o === 0 ? "" : n.substring(0, o)) + n.substring(s + 1) : o = s;
        }
  }
  return n === "" ? null : n;
}
function hn(e, t = !1) {
  var r = t ? " !important;" : ";", n = "";
  for (var i of Object.keys(e)) {
    var a = e[i];
    a != null && a !== "" && (n += " " + i + ": " + a + r);
  }
  return n;
}
function hr(e) {
  return e[0] !== "-" || e[1] !== "-" ? e.toLowerCase() : e;
}
function Va(e, t) {
  if (t) {
    var r = "", n, i;
    if (Array.isArray(t) ? (n = t[0], i = t[1]) : n = t, e) {
      e = String(e).replaceAll(/\s*\/\*.*?\*\/\s*/g, "").trim();
      var a = !1, o = 0, s = !1, u = [];
      n && u.push(...Object.keys(n).map(hr)), i && u.push(...Object.keys(i).map(hr));
      var l = 0, h = -1;
      const _ = e.length;
      for (var c = 0; c < _; c++) {
        var m = e[c];
        if (s ? m === "/" && e[c - 1] === "*" && (s = !1) : a ? a === m && (a = !1) : m === "/" && e[c + 1] === "*" ? s = !0 : m === '"' || m === "'" ? a = m : m === "(" ? o++ : m === ")" && o--, !s && a === !1 && o === 0) {
          if (m === ":" && h === -1)
            h = c;
          else if (m === ";" || c === _ - 1) {
            if (h !== -1) {
              var T = hr(e.substring(l, h).trim());
              if (!u.includes(T)) {
                m !== ";" && c++;
                var d = e.substring(l, c).trim();
                r += " " + d + ";";
              }
            }
            l = c + 1, h = -1;
          }
        }
      }
    }
    return n && (r += hn(n)), i && (r += hn(i, !0)), r = r.trim(), r === "" ? null : r;
  }
  return e == null ? null : String(e);
}
function $e(e, t, r, n, i, a) {
  var o = (
    /** @type {any} */
    e[rn]
  );
  if (o !== r || o === void 0) {
    var s = ja(r, n, a);
    s == null ? e.removeAttribute("class") : t ? e.className = s : e.setAttribute("class", s), e[rn] = r;
  } else if (a && i !== a)
    for (var u in a) {
      var l = !!a[u];
      (i == null || l !== !!i[u]) && e.classList.toggle(u, l);
    }
  return a;
}
function cr(e, t = {}, r, n) {
  for (var i in r) {
    var a = r[i];
    t[i] !== a && (r[i] == null ? e.style.removeProperty(i) : e.style.setProperty(i, a, n));
  }
}
function Ce(e, t, r, n) {
  var i = (
    /** @type {any} */
    e[nn]
  );
  if (i !== t) {
    var a = Va(t, n);
    a == null ? e.removeAttribute("style") : e.style.cssText = a, e[nn] = t;
  } else n && (Array.isArray(n) ? (cr(e, r?.[0], n[0]), cr(e, r?.[1], n[1], "important")) : cr(e, r, n));
  return n;
}
function Pr(e, t, r = !1) {
  if (e.multiple) {
    if (t == null)
      return;
    if (!Gr(t))
      return zi();
    for (var n of e.options)
      n.selected = t.includes(cn(n));
    return;
  }
  for (n of e.options) {
    var i = cn(n);
    if (Xi(i, t)) {
      n.selected = !0;
      return;
    }
  }
  (!r || t !== void 0) && (e.selectedIndex = -1);
}
function za(e) {
  var t = new MutationObserver(() => {
    Pr(e, e.__value);
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
  }), jr(() => {
    t.disconnect();
  });
}
function cn(e) {
  return "__value" in e ? e.__value : e.value;
}
const wt = /* @__PURE__ */ Symbol("class"), at = /* @__PURE__ */ Symbol("style"), Jn = /* @__PURE__ */ Symbol("is custom element"), Qn = /* @__PURE__ */ Symbol("is html"), Xa = Xr ? "input" : "INPUT", Wa = Xr ? "option" : "OPTION", Za = Xr ? "select" : "SELECT";
function qa(e, t) {
  t ? e.hasAttribute("selected") || e.setAttribute("selected", "") : e.removeAttribute("selected");
}
function lt(e, t, r, n) {
  var i = Kn(e);
  i[t] !== (i[t] = r) && (t === "loading" && (e[Wi] = r), r == null ? e.removeAttribute(t) : typeof r != "string" && $n(e).includes(t) ? e[t] = r : e.setAttribute(t, r));
}
function Ya(e, t, r, n, i = !1, a = !1) {
  var o = Kn(e), s = o[Jn], u = !o[Qn], l = t || {}, h = e.nodeName === Wa;
  for (var c in t)
    c in r || (r[c] = null);
  r.class ? r.class = Ga(r.class) : r.class = null, r[at] && (r.style ??= null);
  var m = $n(e);
  if (e.nodeName === Xa && "type" in r && ("value" in r || "__value" in r)) {
    var T = r.type;
    (T !== l.type || T === void 0 && e.hasAttribute("type")) && (l.type = T, lt(e, "type", T));
  }
  for (const E in r) {
    let x = r[E];
    if (h && E === "value" && x == null) {
      e.value = e.__value = "", l[E] = x;
      continue;
    }
    if (E === "class") {
      var d = e.namespaceURI === "http://www.w3.org/1999/xhtml";
      $e(e, d, x, n, t?.[wt], r[wt]), l[E] = x, l[wt] = r[wt];
      continue;
    }
    if (E === "style") {
      Ce(e, x, t?.[at], r[at]), l[E] = x, l[at] = r[at];
      continue;
    }
    var _ = l[E];
    if (!(x === _ && !(x === void 0 && e.hasAttribute(E)))) {
      l[E] = x;
      var P = E[0] + E[1];
      if (P !== "$$")
        if (P === "on") {
          const A = {}, H = "$$" + E;
          let B = E.slice(2);
          var p = ra(B);
          if (Ki(B) && (B = B.slice(0, -7), A.capture = !0), !p && _) {
            if (x != null) continue;
            e.removeEventListener(B, l[H], A), l[H] = null;
          }
          if (p)
            Tt(B, e, x), Ht([B]);
          else if (x != null) {
            let O = function(U) {
              l[E].call(this, U);
            };
            l[H] = $i(B, e, O, A);
          }
        } else if (E === "style")
          lt(e, E, x);
        else if (E === "autofocus")
          Ta(
            /** @type {HTMLElement} */
            e,
            !!x
          );
        else if (!s && (E === "__value" || E === "value" && x != null))
          e.value = e.__value = x;
        else if (E === "selected" && h)
          qa(
            /** @type {HTMLOptionElement} */
            e,
            x
          );
        else {
          var b = E;
          u || (b = ea(b));
          var w = b === "defaultValue" || b === "defaultChecked";
          if (x == null && !s && !w)
            if (o[E] = null, b === "value" || b === "checked") {
              let A = (
                /** @type {HTMLInputElement} */
                e
              );
              const H = t === void 0;
              if (b === "value") {
                let B = A.defaultValue;
                A.removeAttribute(b), A.defaultValue = B, A.value = A.__value = H ? B : null;
              } else {
                let B = A.defaultChecked;
                A.removeAttribute(b), A.defaultChecked = B, A.checked = H ? B : !1;
              }
            } else
              e.removeAttribute(E);
          else w || m.includes(b) && (s || typeof x != "string") ? (e[b] = x, b in o && (o[b] = ta)) : typeof x != "function" && lt(e, b, x);
        }
    }
  }
  return l;
}
function Ja(e, t, r = [], n = [], i = [], a, o = !1, s = !1) {
  Ji(i, r, n, (u) => {
    var l = void 0, h = {}, c = e.nodeName === Za, m = !1;
    if (jn(() => {
      var d = t(...u.map(g)), _ = Ya(
        e,
        l,
        d,
        a,
        o,
        s
      );
      m && c && "value" in d && Pr(
        /** @type {HTMLSelectElement} */
        e,
        d.value
      );
      for (let p of Object.getOwnPropertySymbols(h))
        d[p] || Ke(h[p]);
      for (let p of Object.getOwnPropertySymbols(d)) {
        var P = d[p];
        p.description === Qi && (!l || P !== l[p]) && (h[p] && Ke(h[p]), h[p] = tt(() => Ua(e, () => P))), _[p] = P;
      }
      l = _;
    }), c) {
      var T = (
        /** @type {HTMLSelectElement} */
        e
      );
      Jt(() => {
        Pr(
          T,
          /** @type {Record<string | symbol, any>} */
          l.value,
          !0
        ), za(T);
      });
    }
    m = !0;
  });
}
function Kn(e) {
  return (
    /** @type {Record<string | symbol, unknown>} **/
    /** @type {any} */
    e[Zi] ??= {
      [Jn]: e.nodeName.includes("-"),
      [Qn]: e.namespaceURI === qi
    }
  );
}
var dn = /* @__PURE__ */ new Map();
function $n(e) {
  var t = e.getAttribute("is") || e.nodeName, r = dn.get(t);
  if (r) return r;
  dn.set(t, r = []);
  for (var n, i = e, a = Element.prototype; a !== i; ) {
    n = Yi(i);
    for (var o in n)
      n[o].set && // better safe than sorry, we don't want spread attributes to mess with HTML content
      o !== "innerHTML" && o !== "textContent" && o !== "innerText" && r.push(o);
    i = Ln(i);
  }
  return r;
}
function dr(e, t) {
  return e === t || e?.[Wr] === t;
}
function Xt(e = {}, t, r, n) {
  var i = (
    /** @type {ComponentContext} */
    ot.r
  ), a = (
    /** @type {Effect} */
    qt
  );
  return Jt(() => {
    var o, s;
    return na(() => {
      o = s, s = [], ne(() => {
        dr(r(...s), e) || (t(e, ...s), o && dr(r(...o), e) && t(null, ...o));
      });
    }), () => {
      let u = a;
      for (; u !== i && u.parent !== null && u.parent.f & ia; )
        u = u.parent;
      const l = () => {
        s && dr(r(...s), e) && t(null, ...s);
      }, h = u.teardown;
      u.teardown = () => {
        l(), h?.();
      };
    };
  }), e;
}
function Qa(e = !1) {
  const t = (
    /** @type {ComponentContextLegacy} */
    ot
  ), r = t.l.u;
  if (!r) return;
  let n = () => xe(t.s);
  if (e) {
    let i = 0, a = (
      /** @type {Record<string, any>} */
      {}
    );
    const o = Tr(() => {
      let s = !1;
      const u = t.s;
      for (const l in u)
        u[l] !== a[l] && (a[l] = u[l], s = !0);
      return s && i++, i;
    });
    n = () => g(o);
  }
  r.b.length && aa(() => {
    pn(t, n), wr(r.b);
  }), Ee(() => {
    const i = ne(() => r.m.map(sa));
    return () => {
      for (const a of i)
        typeof a == "function" && a();
    };
  }), r.a.length && Ee(() => {
    pn(t, n), wr(r.a);
  });
}
function pn(e, t) {
  if (e.l.s)
    for (const r of e.l.s) g(r);
  t();
}
const Ka = {
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
function $a(e, t, r) {
  return new Proxy(
    { props: e, exclude: t },
    Ka
  );
}
const es = {
  get(e, t) {
    let r = e.props.length;
    for (; r--; ) {
      let n = e.props[r];
      if (_t(n) && (n = n()), typeof n == "object" && n !== null && t in n) return n[t];
    }
  },
  set(e, t, r) {
    let n = e.props.length;
    for (; n--; ) {
      let i = e.props[n];
      _t(i) && (i = i());
      const a = Sr(i, t);
      if (a && a.set)
        return a.set(r), !0;
    }
    return !1;
  },
  getOwnPropertyDescriptor(e, t) {
    let r = e.props.length;
    for (; r--; ) {
      let n = e.props[r];
      if (_t(n) && (n = n()), typeof n == "object" && n !== null && t in n) {
        const i = Sr(n, t);
        return i && !i.configurable && (i.configurable = !0), i;
      }
    }
  },
  has(e, t) {
    if (t === Wr || t === Vn) return !1;
    for (let r of e.props)
      if (_t(r) && (r = r()), r != null && t in r) return !0;
    return !1;
  },
  ownKeys(e) {
    const t = [];
    for (let r of e.props)
      if (_t(r) && (r = r()), !!r) {
        for (const n in r)
          t.includes(n) || t.push(n);
        for (const n of Object.getOwnPropertySymbols(r))
          t.includes(n) || t.push(n);
      }
    return t;
  }
};
function ts(...e) {
  return new Proxy({ props: e }, es);
}
function I(e, t, r, n) {
  var i = !Un || (r & fa) !== 0, a = (r & ua) !== 0, o = (r & ca) !== 0, s = (
    /** @type {V} */
    n
  ), u = !0, l = (
    /** @type {Derived<V> | undefined} */
    void 0
  ), h = () => o && i ? (l ??= Tr(
    /** @type {() => V} */
    n
  ), g(l)) : (u && (u = !1, s = o ? ne(
    /** @type {() => V} */
    n
  ) : (
    /** @type {V} */
    n
  )), s);
  let c;
  if (a) {
    var m = Wr in e || Vn in e;
    c = Sr(e, t)?.set ?? (m && t in e ? (E) => e[t] = E : void 0);
  }
  var T, d = !1;
  a ? [T, d] = wa(() => (
    /** @type {V} */
    e[t]
  )) : T = /** @type {V} */
  e[t], T === void 0 && n !== void 0 && (T = h(), c && (i && oa(), c(T)));
  var _;
  if (i ? _ = () => {
    var E = (
      /** @type {V} */
      e[t]
    );
    return E === void 0 ? h() : (u = !0, E);
  } : _ = () => {
    var E = (
      /** @type {V} */
      e[t]
    );
    return E !== void 0 && (s = /** @type {V} */
    void 0), E === void 0 ? s : E;
  }, i && (r & la) === 0)
    return _;
  if (c) {
    var P = e.$$legacy;
    return (
      /** @type {() => V} */
      (function(E, x) {
        return arguments.length > 0 ? ((!i || !x || P || d) && c(x ? _() : E), E) : _();
      })
    );
  }
  var p = !1, b = ((r & ha) !== 0 ? Tr : Fn)(() => (p = !1, _()));
  a && g(b);
  var w = (
    /** @type {Effect} */
    qt
  );
  return (
    /** @type {() => V} */
    (function(E, x) {
      if (arguments.length > 0) {
        const A = x ? g(b) : i && a ? zt(E) : E;
        return D(b, A), p = !0, s !== void 0 && (s = A), E;
      }
      return da && p || (w.f & Gn) !== 0 ? b.v : g(b);
    })
  );
}
pa();
var rs = /* @__PURE__ */ Wn('<svg class="resize-handle svelte-vtbxio" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><line x1="1" y1="9" x2="9" y2="1" stroke="gray" stroke-width="0.5" class="svelte-vtbxio"></line><line x1="5" y1="9" x2="9" y2="5" stroke="gray" stroke-width="0.5" class="svelte-vtbxio"></line></svg>'), vn = /* @__PURE__ */ fe("<!> <!>", 1), ns = /* @__PURE__ */ fe('<div class="placeholder svelte-vtbxio"></div>');
function is(e, t) {
  Kt(t, !1);
  let r = I(t, "height", 8, void 0), n = I(t, "min_height", 8, void 0), i = I(t, "max_height", 8, void 0), a = I(t, "width", 8, void 0), o = I(t, "elem_id", 8, ""), s = I(t, "elem_classes", 24, () => []), u = I(t, "variant", 8, "solid"), l = I(t, "border_mode", 8, "base"), h = I(t, "padding", 8, !0), c = I(t, "type", 8, "normal"), m = I(t, "test_id", 8, void 0), T = I(t, "explicit_call", 8, !1), d = I(t, "container", 8, !0), _ = I(t, "visible", 8, !0), P = I(t, "allow_overflow", 8, !0), p = I(t, "overflow_behavior", 8, "auto"), b = I(t, "scale", 8, null), w = I(t, "min_width", 8, 0), E = I(t, "flex", 12, !1), x = I(t, "resizable", 8, !1), A = I(t, "rtl", 8, !1), H = I(t, "fullscreen", 12, !1), B = I(t, "label", 8, void 0), O = Qe(H()), U = Qe(), $ = c() === "fieldset" ? "fieldset" : "div", he = Qe(0), z = Qe(0), G = Qe(null);
  function Ue(ie) {
    H() && ie.key === "Escape" && H(!1);
  }
  const be = (ie) => {
    if (ie !== void 0) {
      if (typeof ie == "number")
        return ie + "px";
      if (typeof ie == "string")
        return ie;
    }
  }, Se = (ie) => {
    let He = ie.clientY;
    const _e = (oe) => {
      const ue = oe.clientY - He;
      He = oe.clientY, ma(U, g(U).style.height = `${g(U).offsetHeight + ue}px`);
    }, Le = () => {
      window.removeEventListener("mousemove", _e), window.removeEventListener("mouseup", Le);
    };
    window.addEventListener("mousemove", _e), window.addEventListener("mouseup", Le);
  };
  an(
    () => (xe(H()), g(O), g(U)),
    () => {
      H() !== g(O) && (D(O, H()), H() ? (D(G, g(U).getBoundingClientRect()), D(he, g(U).offsetHeight), D(z, g(U).offsetWidth), window.addEventListener("keydown", Ue)) : (D(G, null), window.removeEventListener("keydown", Ue)));
    }
  ), an(() => xe(_()), () => {
    _() || E(!1);
  }), va(), Qa();
  var Ae = st(), Fe = pe(Ae);
  {
    var ce = (ie) => {
      var He = vn(), _e = pe(He);
      Da(_e, () => $, !1, (ue, Ie) => {
        Xt(ue, (ae) => D(U, ae), () => g(U)), Ja(
          ue,
          (ae, Re) => ({
            "data-testid": m(),
            id: o(),
            class: `block ${ae ?? ""}`,
            dir: A() ? "rtl" : "ltr",
            "aria-label": B(),
            style: "",
            [wt]: {
              hidden: _() === "hidden",
              padded: h(),
              flex: E(),
              border_focus: l() === "focus",
              border_contrast: l() === "contrast",
              "hide-container": !T() && !d(),
              fullscreen: H(),
              animating: H() && g(G) !== null,
              "auto-margin": b() === null
            },
            [at]: Re
          }),
          [
            () => (xe(s()), ne(() => s()?.join(" ") || "")),
            () => ({
              height: (xe(H()), xe(r()), ne(() => H() ? void 0 : be(r()))),
              "min-height": (xe(H()), xe(n()), ne(() => H() ? void 0 : be(n()))),
              "max-height": (xe(H()), xe(i()), ne(() => H() ? void 0 : be(i()))),
              "--start-top": (g(G), ne(() => g(G) ? `${g(G).top}px` : "0px")),
              "--start-left": (g(G), ne(() => g(G) ? `${g(G).left}px` : "0px")),
              "--start-width": (g(G), ne(() => g(G) ? `${g(G).width}px` : "0px")),
              "--start-height": (g(G), ne(() => g(G) ? `${g(G).height}px` : "0px")),
              width: (xe(H()), xe(a()), ne(() => H() ? void 0 : typeof a() == "number" ? `calc(min(${a()}px, 100%))` : be(a()))),
              "border-style": u(),
              overflow: P() ? p() : "hidden",
              "flex-grow": b(),
              "min-width": `calc(min(${w()}px, 100%))`
            })
          ],
          void 0,
          void 0,
          "svelte-vtbxio"
        );
        var Ne = vn(), Ge = pe(Ne);
        Br(Ge, t, "default", {});
        var Be = Z(Ge, 2);
        {
          var Ze = (ae) => {
            var Re = rs();
            Ar("mousedown", Re, Se), N(ae, Re);
          };
          Y(Be, (ae) => {
            x() && ae(Ze);
          });
        }
        N(Ie, Ne);
      });
      var Le = Z(_e, 2);
      {
        var oe = (ue) => {
          var Ie = ns();
          let Ne;
          K(() => Ne = Ce(Ie, "", Ne, {
            height: g(he) + "px",
            width: g(z) + "px"
          })), N(ue, Ie);
        };
        Y(Le, (ue) => {
          H() && ue(oe);
        });
      }
      N(ie, He);
    };
    Y(Fe, (ie) => {
      (_() === !0 || _() === "hidden") && ie(ce);
    });
  }
  N(e, Ae), Qt();
}
var as = /* @__PURE__ */ fe('<span class="svelte-7nlo3u"> </span>'), ss = /* @__PURE__ */ fe("<button><!> <div><!> <!></div></button>");
function mn(e, t) {
  let r = I(t, "label", 3, ""), n = I(t, "show_label", 3, !1), i = I(t, "pending", 3, !1), a = I(t, "size", 3, "small"), o = I(t, "padded", 3, !0), s = I(t, "highlight", 3, !1), u = I(t, "disabled", 3, !1), l = I(t, "hasPopup", 3, !1), h = I(t, "color", 3, "var(--block-label-text-color)"), c = I(t, "transparent", 3, !1), m = I(t, "background", 3, "var(--block-background-fill)"), T = I(t, "border", 3, "transparent"), d = Oe(() => s() ? "var(--color-accent)" : h());
  var _ = ss();
  let P, p;
  var b = se(_);
  {
    var w = (O) => {
      var U = as(), $ = se(U);
      K(() => ge($, r())), N(O, U);
    };
    Y(b, (O) => {
      n() && O(w);
    });
  }
  var E = Z(b, 2);
  let x;
  var A = se(E);
  La(A, () => t.Icon, (O, U) => {
    U(O, {});
  });
  var H = Z(A, 2);
  {
    var B = (O) => {
      var U = st(), $ = pe(U);
      Ia($, () => t.children), N(O, U);
    };
    Y(H, (O) => {
      t.children && O(B);
    });
  }
  K(() => {
    P = $e(_, 1, "icon-button svelte-7nlo3u", null, P, {
      pending: i(),
      padded: o(),
      highlight: s(),
      transparent: c()
    }), _.disabled = u(), lt(_, "aria-label", r()), lt(_, "aria-haspopup", l()), lt(_, "title", r()), p = Ce(_, "", p, {
      "--border-color": T(),
      color: !u() && g(d) ? g(d) : "var(--block-label-text-color)",
      "--bg-color": u() ? "auto" : m()
    }), x = $e(E, 1, "svelte-7nlo3u", null, x, {
      "x-small": a() === "x-small",
      small: a() === "small",
      large: a() === "large",
      medium: a() === "medium"
    });
  }), Tt("click", _, function(...O) {
    t.onclick?.apply(this, O);
  }), N(e, _);
}
Ht(["click"]);
var os = /* @__PURE__ */ Wn('<svg width="100%" height="100%" viewBox="0 0 24 24" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:space="preserve" stroke="currentColor" style="fill-rule:evenodd;clip-rule:evenodd;stroke-linecap:round;stroke-linejoin:round;"><g transform="matrix(1.14096,-0.140958,-0.140958,1.14096,-0.0559523,0.0559523)"><path d="M18,6L6.087,17.913" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></g><path d="M4.364,4.364L19.636,19.636" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></svg>');
function gn(e) {
  var t = os();
  N(e, t);
}
const ls = [
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
], bn = {
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
ls.reduce((e, { color: t, primary: r, secondary: n }) => ({
  ...e,
  [t]: {
    primary: bn[t][r],
    secondary: bn[t][n]
  }
}), {});
function us(e) {
  return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, "default") ? e.default : e;
}
var pr, _n;
function fs() {
  if (_n) return pr;
  _n = 1;
  var e = function(b) {
    return t(b) && !r(b);
  };
  function t(p) {
    return !!p && typeof p == "object";
  }
  function r(p) {
    var b = Object.prototype.toString.call(p);
    return b === "[object RegExp]" || b === "[object Date]" || a(p);
  }
  var n = typeof Symbol == "function" && Symbol.for, i = n ? /* @__PURE__ */ Symbol.for("react.element") : 60103;
  function a(p) {
    return p.$$typeof === i;
  }
  function o(p) {
    return Array.isArray(p) ? [] : {};
  }
  function s(p, b) {
    return b.clone !== !1 && b.isMergeableObject(p) ? _(o(p), p, b) : p;
  }
  function u(p, b, w) {
    return p.concat(b).map(function(E) {
      return s(E, w);
    });
  }
  function l(p, b) {
    if (!b.customMerge)
      return _;
    var w = b.customMerge(p);
    return typeof w == "function" ? w : _;
  }
  function h(p) {
    return Object.getOwnPropertySymbols ? Object.getOwnPropertySymbols(p).filter(function(b) {
      return Object.propertyIsEnumerable.call(p, b);
    }) : [];
  }
  function c(p) {
    return Object.keys(p).concat(h(p));
  }
  function m(p, b) {
    try {
      return b in p;
    } catch {
      return !1;
    }
  }
  function T(p, b) {
    return m(p, b) && !(Object.hasOwnProperty.call(p, b) && Object.propertyIsEnumerable.call(p, b));
  }
  function d(p, b, w) {
    var E = {};
    return w.isMergeableObject(p) && c(p).forEach(function(x) {
      E[x] = s(p[x], w);
    }), c(b).forEach(function(x) {
      T(p, x) || (m(p, x) && w.isMergeableObject(b[x]) ? E[x] = l(x, w)(p[x], b[x], w) : E[x] = s(b[x], w));
    }), E;
  }
  function _(p, b, w) {
    w = w || {}, w.arrayMerge = w.arrayMerge || u, w.isMergeableObject = w.isMergeableObject || e, w.cloneUnlessOtherwiseSpecified = s;
    var E = Array.isArray(b), x = Array.isArray(p), A = E === x;
    return A ? E ? w.arrayMerge(p, b, w) : d(p, b, w) : s(b, w);
  }
  _.all = function(b, w) {
    if (!Array.isArray(b))
      throw new Error("first argument should be an array");
    return b.reduce(function(E, x) {
      return _(E, x, w);
    }, {});
  };
  var P = _;
  return pr = P, pr;
}
var hs = fs();
const cs = /* @__PURE__ */ us(hs);
var Mr = function(e, t) {
  return Mr = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(r, n) {
    r.__proto__ = n;
  } || function(r, n) {
    for (var i in n) Object.prototype.hasOwnProperty.call(n, i) && (r[i] = n[i]);
  }, Mr(e, t);
};
function er(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Class extends value " + String(t) + " is not a constructor or null");
  Mr(e, t);
  function r() {
    this.constructor = e;
  }
  e.prototype = t === null ? Object.create(t) : (r.prototype = t.prototype, new r());
}
var k = function() {
  return k = Object.assign || function(t) {
    for (var r, n = 1, i = arguments.length; n < i; n++) {
      r = arguments[n];
      for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (t[a] = r[a]);
    }
    return t;
  }, k.apply(this, arguments);
};
function ds(e, t) {
  var r = {};
  for (var n in e) Object.prototype.hasOwnProperty.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && typeof Object.getOwnPropertySymbols == "function")
    for (var i = 0, n = Object.getOwnPropertySymbols(e); i < n.length; i++)
      t.indexOf(n[i]) < 0 && Object.prototype.propertyIsEnumerable.call(e, n[i]) && (r[n[i]] = e[n[i]]);
  return r;
}
function vr(e, t, r) {
  if (r || arguments.length === 2) for (var n = 0, i = t.length, a; n < i; n++)
    (a || !(n in t)) && (a || (a = Array.prototype.slice.call(t, 0, n)), a[n] = t[n]);
  return e.concat(a || Array.prototype.slice.call(t));
}
function mr(e, t) {
  var r = t && t.cache ? t.cache : ys, n = t && t.serializer ? t.serializer : bs, i = t && t.strategy ? t.strategy : ms;
  return i(e, {
    cache: r,
    serializer: n
  });
}
function ps(e) {
  return e == null || typeof e == "number" || typeof e == "boolean";
}
function vs(e, t, r, n) {
  var i = ps(n) ? n : r(n), a = t.get(i);
  return typeof a > "u" && (a = e.call(this, n), t.set(i, a)), a;
}
function ei(e, t, r) {
  var n = Array.prototype.slice.call(arguments, 3), i = r(n), a = t.get(i);
  return typeof a > "u" && (a = e.apply(this, n), t.set(i, a)), a;
}
function ti(e, t, r, n, i) {
  return r.bind(t, e, n, i);
}
function ms(e, t) {
  var r = e.length === 1 ? vs : ei;
  return ti(e, this, r, t.cache.create(), t.serializer);
}
function gs(e, t) {
  return ti(e, this, ei, t.cache.create(), t.serializer);
}
var bs = function() {
  return JSON.stringify(arguments);
}, _s = (
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
), ys = {
  create: function() {
    return new _s();
  }
}, gr = {
  variadic: gs
}, L;
(function(e) {
  e[e.EXPECT_ARGUMENT_CLOSING_BRACE = 1] = "EXPECT_ARGUMENT_CLOSING_BRACE", e[e.EMPTY_ARGUMENT = 2] = "EMPTY_ARGUMENT", e[e.MALFORMED_ARGUMENT = 3] = "MALFORMED_ARGUMENT", e[e.EXPECT_ARGUMENT_TYPE = 4] = "EXPECT_ARGUMENT_TYPE", e[e.INVALID_ARGUMENT_TYPE = 5] = "INVALID_ARGUMENT_TYPE", e[e.EXPECT_ARGUMENT_STYLE = 6] = "EXPECT_ARGUMENT_STYLE", e[e.INVALID_NUMBER_SKELETON = 7] = "INVALID_NUMBER_SKELETON", e[e.INVALID_DATE_TIME_SKELETON = 8] = "INVALID_DATE_TIME_SKELETON", e[e.EXPECT_NUMBER_SKELETON = 9] = "EXPECT_NUMBER_SKELETON", e[e.EXPECT_DATE_TIME_SKELETON = 10] = "EXPECT_DATE_TIME_SKELETON", e[e.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE = 11] = "UNCLOSED_QUOTE_IN_ARGUMENT_STYLE", e[e.EXPECT_SELECT_ARGUMENT_OPTIONS = 12] = "EXPECT_SELECT_ARGUMENT_OPTIONS", e[e.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE = 13] = "EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE = 14] = "INVALID_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR = 15] = "EXPECT_SELECT_ARGUMENT_SELECTOR", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR = 16] = "EXPECT_PLURAL_ARGUMENT_SELECTOR", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT = 17] = "EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT = 18] = "EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT", e[e.INVALID_PLURAL_ARGUMENT_SELECTOR = 19] = "INVALID_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_PLURAL_ARGUMENT_SELECTOR = 20] = "DUPLICATE_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_SELECT_ARGUMENT_SELECTOR = 21] = "DUPLICATE_SELECT_ARGUMENT_SELECTOR", e[e.MISSING_OTHER_CLAUSE = 22] = "MISSING_OTHER_CLAUSE", e[e.INVALID_TAG = 23] = "INVALID_TAG", e[e.INVALID_TAG_NAME = 25] = "INVALID_TAG_NAME", e[e.UNMATCHED_CLOSING_TAG = 26] = "UNMATCHED_CLOSING_TAG", e[e.UNCLOSED_TAG = 27] = "UNCLOSED_TAG";
})(L || (L = {}));
var V;
(function(e) {
  e[e.literal = 0] = "literal", e[e.argument = 1] = "argument", e[e.number = 2] = "number", e[e.date = 3] = "date", e[e.time = 4] = "time", e[e.select = 5] = "select", e[e.plural = 6] = "plural", e[e.pound = 7] = "pound", e[e.tag = 8] = "tag";
})(V || (V = {}));
var ft;
(function(e) {
  e[e.number = 0] = "number", e[e.dateTime = 1] = "dateTime";
})(ft || (ft = {}));
function yn(e) {
  return e.type === V.literal;
}
function xs(e) {
  return e.type === V.argument;
}
function ri(e) {
  return e.type === V.number;
}
function ni(e) {
  return e.type === V.date;
}
function ii(e) {
  return e.type === V.time;
}
function ai(e) {
  return e.type === V.select;
}
function si(e) {
  return e.type === V.plural;
}
function Es(e) {
  return e.type === V.pound;
}
function oi(e) {
  return e.type === V.tag;
}
function li(e) {
  return !!(e && typeof e == "object" && e.type === ft.number);
}
function Or(e) {
  return !!(e && typeof e == "object" && e.type === ft.dateTime);
}
var ui = /[ \xA0\u1680\u2000-\u200A\u202F\u205F\u3000]/, ws = /(?:[Eec]{1,6}|G{1,5}|[Qq]{1,5}|(?:[yYur]+|U{1,5})|[ML]{1,5}|d{1,2}|D{1,3}|F{1}|[abB]{1,5}|[hkHK]{1,2}|w{1,2}|W{1}|m{1,2}|s{1,2}|[zZOvVxX]{1,4})(?=([^']*'[^']*')*[^']*$)/g;
function Ts(e) {
  var t = {};
  return e.replace(ws, function(r) {
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
var Ss = /[\t-\r \x85\u200E\u200F\u2028\u2029]/i;
function As(e) {
  if (e.length === 0)
    throw new Error("Number skeleton cannot be empty");
  for (var t = e.split(Ss).filter(function(m) {
    return m.length > 0;
  }), r = [], n = 0, i = t; n < i.length; n++) {
    var a = i[n], o = a.split("/");
    if (o.length === 0)
      throw new Error("Invalid number skeleton");
    for (var s = o[0], u = o.slice(1), l = 0, h = u; l < h.length; l++) {
      var c = h[l];
      if (c.length === 0)
        throw new Error("Invalid number skeleton");
    }
    r.push({ stem: s, options: u });
  }
  return r;
}
function Hs(e) {
  return e.replace(/^(.*?)-/, "");
}
var xn = /^\.(?:(0+)(\*)?|(#+)|(0+)(#+))$/g, fi = /^(@+)?(\+|#+)?[rs]?$/g, Is = /(\*)(0+)|(#+)(0+)|(0+)/g, hi = /^(0+)$/;
function En(e) {
  var t = {};
  return e[e.length - 1] === "r" ? t.roundingPriority = "morePrecision" : e[e.length - 1] === "s" && (t.roundingPriority = "lessPrecision"), e.replace(fi, function(r, n, i) {
    return typeof i != "string" ? (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length) : i === "+" ? t.minimumSignificantDigits = n.length : n[0] === "#" ? t.maximumSignificantDigits = n.length : (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length + (typeof i == "string" ? i.length : 0)), "";
  }), t;
}
function ci(e) {
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
function Bs(e) {
  var t;
  if (e[0] === "E" && e[1] === "E" ? (t = {
    notation: "engineering"
  }, e = e.slice(2)) : e[0] === "E" && (t = {
    notation: "scientific"
  }, e = e.slice(1)), t) {
    var r = e.slice(0, 2);
    if (r === "+!" ? (t.signDisplay = "always", e = e.slice(2)) : r === "+?" && (t.signDisplay = "exceptZero", e = e.slice(2)), !hi.test(e))
      throw new Error("Malformed concise eng/scientific notation");
    t.minimumIntegerDigits = e.length;
  }
  return t;
}
function wn(e) {
  var t = {}, r = ci(e);
  return r || t;
}
function Ps(e) {
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
        t.style = "unit", t.unit = Hs(i.options[0]);
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
        t = k(k(k({}, t), { notation: "scientific" }), i.options.reduce(function(u, l) {
          return k(k({}, u), wn(l));
        }, {}));
        continue;
      case "engineering":
        t = k(k(k({}, t), { notation: "engineering" }), i.options.reduce(function(u, l) {
          return k(k({}, u), wn(l));
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
        i.options[0].replace(Is, function(u, l, h, c, m, T) {
          if (l)
            t.minimumIntegerDigits = h.length;
          else {
            if (c && m)
              throw new Error("We currently do not support maximum integer digits");
            if (T)
              throw new Error("We currently do not support exact integer digits");
          }
          return "";
        });
        continue;
    }
    if (hi.test(i.stem)) {
      t.minimumIntegerDigits = i.stem.length;
      continue;
    }
    if (xn.test(i.stem)) {
      if (i.options.length > 1)
        throw new RangeError("Fraction-precision stems only accept a single optional option");
      i.stem.replace(xn, function(u, l, h, c, m, T) {
        return h === "*" ? t.minimumFractionDigits = l.length : c && c[0] === "#" ? t.maximumFractionDigits = c.length : m && T ? (t.minimumFractionDigits = m.length, t.maximumFractionDigits = m.length + T.length) : (t.minimumFractionDigits = l.length, t.maximumFractionDigits = l.length), "";
      });
      var a = i.options[0];
      a === "w" ? t = k(k({}, t), { trailingZeroDisplay: "stripIfInteger" }) : a && (t = k(k({}, t), En(a)));
      continue;
    }
    if (fi.test(i.stem)) {
      t = k(k({}, t), En(i.stem));
      continue;
    }
    var o = ci(i.stem);
    o && (t = k(k({}, t), o));
    var s = Bs(i.stem);
    s && (t = k(k({}, t), s));
  }
  return t;
}
var Dt = {
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
function Ms(e, t) {
  for (var r = "", n = 0; n < e.length; n++) {
    var i = e.charAt(n);
    if (i === "j") {
      for (var a = 0; n + 1 < e.length && e.charAt(n + 1) === i; )
        a++, n++;
      var o = 1 + (a & 1), s = a < 2 ? 1 : 3 + (a >> 1), u = "a", l = Os(t);
      for ((l == "H" || l == "k") && (s = 0); s-- > 0; )
        r += u;
      for (; o-- > 0; )
        r = l + r;
    } else i === "J" ? r += "H" : r += i;
  }
  return r;
}
function Os(e) {
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
  var i = Dt[n || ""] || Dt[r || ""] || Dt["".concat(r, "-001")] || Dt["001"];
  return i[0];
}
var br, Cs = new RegExp("^".concat(ui.source, "*")), Ls = new RegExp("".concat(ui.source, "*$"));
function R(e, t) {
  return { start: e, end: t };
}
var Ns = !!String.prototype.startsWith && "_a".startsWith("a", 1), Rs = !!String.fromCodePoint, Ds = !!Object.fromEntries, ks = !!String.prototype.codePointAt, Us = !!String.prototype.trimStart, Fs = !!String.prototype.trimEnd, Gs = !!Number.isSafeInteger, js = Gs ? Number.isSafeInteger : function(e) {
  return typeof e == "number" && isFinite(e) && Math.floor(e) === e && Math.abs(e) <= 9007199254740991;
}, Cr = !0;
try {
  var Vs = pi("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  Cr = ((br = Vs.exec("a")) === null || br === void 0 ? void 0 : br[0]) === "a";
} catch {
  Cr = !1;
}
var Tn = Ns ? (
  // Native
  function(t, r, n) {
    return t.startsWith(r, n);
  }
) : (
  // For IE11
  function(t, r, n) {
    return t.slice(n, n + r.length) === r;
  }
), Lr = Rs ? String.fromCodePoint : (
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
), Sn = (
  // native
  Ds ? Object.fromEntries : (
    // Ponyfill
    function(t) {
      for (var r = {}, n = 0, i = t; n < i.length; n++) {
        var a = i[n], o = a[0], s = a[1];
        r[o] = s;
      }
      return r;
    }
  )
), di = ks ? (
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
), zs = Us ? (
  // Native
  function(t) {
    return t.trimStart();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(Cs, "");
  }
), Xs = Fs ? (
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
function pi(e, t) {
  return new RegExp(e, t);
}
var Nr;
if (Cr) {
  var An = pi("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  Nr = function(t, r) {
    var n;
    An.lastIndex = r;
    var i = An.exec(t);
    return (n = i[1]) !== null && n !== void 0 ? n : "";
  };
} else
  Nr = function(t, r) {
    for (var n = []; ; ) {
      var i = di(t, r);
      if (i === void 0 || vi(i) || Ys(i))
        break;
      n.push(i), r += i >= 65536 ? 2 : 1;
    }
    return Lr.apply(void 0, n);
  };
var Ws = (
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
              type: V.pound,
              location: R(s, this.clonePosition())
            });
          } else if (a === 60 && !this.ignoreTag && this.peek() === 47) {
            if (n)
              break;
            return this.error(L.UNMATCHED_CLOSING_TAG, R(this.clonePosition(), this.clonePosition()));
          } else if (a === 60 && !this.ignoreTag && Rr(this.peek() || 0)) {
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
            type: V.literal,
            value: "<".concat(i, "/>"),
            location: R(n, this.clonePosition())
          },
          err: null
        };
      if (this.bumpIf(">")) {
        var a = this.parseMessage(t + 1, r, !0);
        if (a.err)
          return a;
        var o = a.val, s = this.clonePosition();
        if (this.bumpIf("</")) {
          if (this.isEOF() || !Rr(this.char()))
            return this.error(L.INVALID_TAG, R(s, this.clonePosition()));
          var u = this.clonePosition(), l = this.parseTagName();
          return i !== l ? this.error(L.UNMATCHED_CLOSING_TAG, R(u, this.clonePosition())) : (this.bumpSpace(), this.bumpIf(">") ? {
            val: {
              type: V.tag,
              value: i,
              children: o,
              location: R(n, this.clonePosition())
            },
            err: null
          } : this.error(L.INVALID_TAG, R(s, this.clonePosition())));
        } else
          return this.error(L.UNCLOSED_TAG, R(n, this.clonePosition()));
      } else
        return this.error(L.INVALID_TAG, R(n, this.clonePosition()));
    }, e.prototype.parseTagName = function() {
      var t = this.offset();
      for (this.bump(); !this.isEOF() && qs(this.char()); )
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
      var u = R(n, this.clonePosition());
      return {
        val: { type: V.literal, value: i, location: u },
        err: null
      };
    }, e.prototype.tryParseLeftAngleBracket = function() {
      return !this.isEOF() && this.char() === 60 && (this.ignoreTag || // If at the opening tag or closing tag position, bail.
      !Zs(this.peek() || 0)) ? (this.bump(), "<") : null;
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
      return Lr.apply(void 0, r);
    }, e.prototype.tryParseUnquoted = function(t, r) {
      if (this.isEOF())
        return null;
      var n = this.char();
      return n === 60 || n === 123 || n === 35 && (r === "plural" || r === "selectordinal") || n === 125 && t > 0 ? null : (this.bump(), Lr(n));
    }, e.prototype.parseArgument = function(t, r) {
      var n = this.clonePosition();
      if (this.bump(), this.bumpSpace(), this.isEOF())
        return this.error(L.EXPECT_ARGUMENT_CLOSING_BRACE, R(n, this.clonePosition()));
      if (this.char() === 125)
        return this.bump(), this.error(L.EMPTY_ARGUMENT, R(n, this.clonePosition()));
      var i = this.parseIdentifierIfPossible().value;
      if (!i)
        return this.error(L.MALFORMED_ARGUMENT, R(n, this.clonePosition()));
      if (this.bumpSpace(), this.isEOF())
        return this.error(L.EXPECT_ARGUMENT_CLOSING_BRACE, R(n, this.clonePosition()));
      switch (this.char()) {
        // Simple argument: `{name}`
        case 125:
          return this.bump(), {
            val: {
              type: V.argument,
              // value does not include the opening and closing braces.
              value: i,
              location: R(n, this.clonePosition())
            },
            err: null
          };
        // Argument with options: `{name, format, ...}`
        case 44:
          return this.bump(), this.bumpSpace(), this.isEOF() ? this.error(L.EXPECT_ARGUMENT_CLOSING_BRACE, R(n, this.clonePosition())) : this.parseArgumentOptions(t, r, i, n);
        default:
          return this.error(L.MALFORMED_ARGUMENT, R(n, this.clonePosition()));
      }
    }, e.prototype.parseIdentifierIfPossible = function() {
      var t = this.clonePosition(), r = this.offset(), n = Nr(this.message, r), i = r + n.length;
      this.bumpTo(i);
      var a = this.clonePosition(), o = R(t, a);
      return { value: n, location: o };
    }, e.prototype.parseArgumentOptions = function(t, r, n, i) {
      var a, o = this.clonePosition(), s = this.parseIdentifierIfPossible().value, u = this.clonePosition();
      switch (s) {
        case "":
          return this.error(L.EXPECT_ARGUMENT_TYPE, R(o, u));
        case "number":
        case "date":
        case "time": {
          this.bumpSpace();
          var l = null;
          if (this.bumpIf(",")) {
            this.bumpSpace();
            var h = this.clonePosition(), c = this.parseSimpleArgStyleIfPossible();
            if (c.err)
              return c;
            var m = Xs(c.val);
            if (m.length === 0)
              return this.error(L.EXPECT_ARGUMENT_STYLE, R(this.clonePosition(), this.clonePosition()));
            var T = R(h, this.clonePosition());
            l = { style: m, styleLocation: T };
          }
          var d = this.tryParseArgumentClose(i);
          if (d.err)
            return d;
          var _ = R(i, this.clonePosition());
          if (l && Tn(l?.style, "::", 0)) {
            var P = zs(l.style.slice(2));
            if (s === "number") {
              var c = this.parseNumberSkeletonFromString(P, l.styleLocation);
              return c.err ? c : {
                val: { type: V.number, value: n, location: _, style: c.val },
                err: null
              };
            } else {
              if (P.length === 0)
                return this.error(L.EXPECT_DATE_TIME_SKELETON, _);
              var p = P;
              this.locale && (p = Ms(P, this.locale));
              var m = {
                type: ft.dateTime,
                pattern: p,
                location: l.styleLocation,
                parsedOptions: this.shouldParseSkeletons ? Ts(p) : {}
              }, b = s === "date" ? V.date : V.time;
              return {
                val: { type: b, value: n, location: _, style: m },
                err: null
              };
            }
          }
          return {
            val: {
              type: s === "number" ? V.number : s === "date" ? V.date : V.time,
              value: n,
              location: _,
              style: (a = l?.style) !== null && a !== void 0 ? a : null
            },
            err: null
          };
        }
        case "plural":
        case "selectordinal":
        case "select": {
          var w = this.clonePosition();
          if (this.bumpSpace(), !this.bumpIf(","))
            return this.error(L.EXPECT_SELECT_ARGUMENT_OPTIONS, R(w, k({}, w)));
          this.bumpSpace();
          var E = this.parseIdentifierIfPossible(), x = 0;
          if (s !== "select" && E.value === "offset") {
            if (!this.bumpIf(":"))
              return this.error(L.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, R(this.clonePosition(), this.clonePosition()));
            this.bumpSpace();
            var c = this.tryParseDecimalInteger(L.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, L.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE);
            if (c.err)
              return c;
            this.bumpSpace(), E = this.parseIdentifierIfPossible(), x = c.val;
          }
          var A = this.tryParsePluralOrSelectOptions(t, s, r, E);
          if (A.err)
            return A;
          var d = this.tryParseArgumentClose(i);
          if (d.err)
            return d;
          var H = R(i, this.clonePosition());
          return s === "select" ? {
            val: {
              type: V.select,
              value: n,
              options: Sn(A.val),
              location: H
            },
            err: null
          } : {
            val: {
              type: V.plural,
              value: n,
              options: Sn(A.val),
              offset: x,
              pluralType: s === "plural" ? "cardinal" : "ordinal",
              location: H
            },
            err: null
          };
        }
        default:
          return this.error(L.INVALID_ARGUMENT_TYPE, R(o, u));
      }
    }, e.prototype.tryParseArgumentClose = function(t) {
      return this.isEOF() || this.char() !== 125 ? this.error(L.EXPECT_ARGUMENT_CLOSING_BRACE, R(t, this.clonePosition())) : (this.bump(), { val: !0, err: null });
    }, e.prototype.parseSimpleArgStyleIfPossible = function() {
      for (var t = 0, r = this.clonePosition(); !this.isEOF(); ) {
        var n = this.char();
        switch (n) {
          case 39: {
            this.bump();
            var i = this.clonePosition();
            if (!this.bumpUntil("'"))
              return this.error(L.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE, R(i, this.clonePosition()));
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
        n = As(t);
      } catch {
        return this.error(L.INVALID_NUMBER_SKELETON, r);
      }
      return {
        val: {
          type: ft.number,
          tokens: n,
          location: r,
          parsedOptions: this.shouldParseSkeletons ? Ps(n) : {}
        },
        err: null
      };
    }, e.prototype.tryParsePluralOrSelectOptions = function(t, r, n, i) {
      for (var a, o = !1, s = [], u = /* @__PURE__ */ new Set(), l = i.value, h = i.location; ; ) {
        if (l.length === 0) {
          var c = this.clonePosition();
          if (r !== "select" && this.bumpIf("=")) {
            var m = this.tryParseDecimalInteger(L.EXPECT_PLURAL_ARGUMENT_SELECTOR, L.INVALID_PLURAL_ARGUMENT_SELECTOR);
            if (m.err)
              return m;
            h = R(c, this.clonePosition()), l = this.message.slice(c.offset, this.offset());
          } else
            break;
        }
        if (u.has(l))
          return this.error(r === "select" ? L.DUPLICATE_SELECT_ARGUMENT_SELECTOR : L.DUPLICATE_PLURAL_ARGUMENT_SELECTOR, h);
        l === "other" && (o = !0), this.bumpSpace();
        var T = this.clonePosition();
        if (!this.bumpIf("{"))
          return this.error(r === "select" ? L.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT : L.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT, R(this.clonePosition(), this.clonePosition()));
        var d = this.parseMessage(t + 1, r, n);
        if (d.err)
          return d;
        var _ = this.tryParseArgumentClose(T);
        if (_.err)
          return _;
        s.push([
          l,
          {
            value: d.val,
            location: R(T, this.clonePosition())
          }
        ]), u.add(l), this.bumpSpace(), a = this.parseIdentifierIfPossible(), l = a.value, h = a.location;
      }
      return s.length === 0 ? this.error(r === "select" ? L.EXPECT_SELECT_ARGUMENT_SELECTOR : L.EXPECT_PLURAL_ARGUMENT_SELECTOR, R(this.clonePosition(), this.clonePosition())) : this.requiresOtherClause && !o ? this.error(L.MISSING_OTHER_CLAUSE, R(this.clonePosition(), this.clonePosition())) : { val: s, err: null };
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
      var u = R(i, this.clonePosition());
      return a ? (o *= n, js(o) ? { val: o, err: null } : this.error(r, u)) : this.error(t, u);
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
      var r = di(this.message, t);
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
      if (Tn(this.message, t, this.offset())) {
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
      for (; !this.isEOF() && vi(this.char()); )
        this.bump();
    }, e.prototype.peek = function() {
      if (this.isEOF())
        return null;
      var t = this.char(), r = this.offset(), n = this.message.charCodeAt(r + (t >= 65536 ? 2 : 1));
      return n ?? null;
    }, e;
  })()
);
function Rr(e) {
  return e >= 97 && e <= 122 || e >= 65 && e <= 90;
}
function Zs(e) {
  return Rr(e) || e === 47;
}
function qs(e) {
  return e === 45 || e === 46 || e >= 48 && e <= 57 || e === 95 || e >= 97 && e <= 122 || e >= 65 && e <= 90 || e == 183 || e >= 192 && e <= 214 || e >= 216 && e <= 246 || e >= 248 && e <= 893 || e >= 895 && e <= 8191 || e >= 8204 && e <= 8205 || e >= 8255 && e <= 8256 || e >= 8304 && e <= 8591 || e >= 11264 && e <= 12271 || e >= 12289 && e <= 55295 || e >= 63744 && e <= 64975 || e >= 65008 && e <= 65533 || e >= 65536 && e <= 983039;
}
function vi(e) {
  return e >= 9 && e <= 13 || e === 32 || e === 133 || e >= 8206 && e <= 8207 || e === 8232 || e === 8233;
}
function Ys(e) {
  return e >= 33 && e <= 35 || e === 36 || e >= 37 && e <= 39 || e === 40 || e === 41 || e === 42 || e === 43 || e === 44 || e === 45 || e >= 46 && e <= 47 || e >= 58 && e <= 59 || e >= 60 && e <= 62 || e >= 63 && e <= 64 || e === 91 || e === 92 || e === 93 || e === 94 || e === 96 || e === 123 || e === 124 || e === 125 || e === 126 || e === 161 || e >= 162 && e <= 165 || e === 166 || e === 167 || e === 169 || e === 171 || e === 172 || e === 174 || e === 176 || e === 177 || e === 182 || e === 187 || e === 191 || e === 215 || e === 247 || e >= 8208 && e <= 8213 || e >= 8214 && e <= 8215 || e === 8216 || e === 8217 || e === 8218 || e >= 8219 && e <= 8220 || e === 8221 || e === 8222 || e === 8223 || e >= 8224 && e <= 8231 || e >= 8240 && e <= 8248 || e === 8249 || e === 8250 || e >= 8251 && e <= 8254 || e >= 8257 && e <= 8259 || e === 8260 || e === 8261 || e === 8262 || e >= 8263 && e <= 8273 || e === 8274 || e === 8275 || e >= 8277 && e <= 8286 || e >= 8592 && e <= 8596 || e >= 8597 && e <= 8601 || e >= 8602 && e <= 8603 || e >= 8604 && e <= 8607 || e === 8608 || e >= 8609 && e <= 8610 || e === 8611 || e >= 8612 && e <= 8613 || e === 8614 || e >= 8615 && e <= 8621 || e === 8622 || e >= 8623 && e <= 8653 || e >= 8654 && e <= 8655 || e >= 8656 && e <= 8657 || e === 8658 || e === 8659 || e === 8660 || e >= 8661 && e <= 8691 || e >= 8692 && e <= 8959 || e >= 8960 && e <= 8967 || e === 8968 || e === 8969 || e === 8970 || e === 8971 || e >= 8972 && e <= 8991 || e >= 8992 && e <= 8993 || e >= 8994 && e <= 9e3 || e === 9001 || e === 9002 || e >= 9003 && e <= 9083 || e === 9084 || e >= 9085 && e <= 9114 || e >= 9115 && e <= 9139 || e >= 9140 && e <= 9179 || e >= 9180 && e <= 9185 || e >= 9186 && e <= 9254 || e >= 9255 && e <= 9279 || e >= 9280 && e <= 9290 || e >= 9291 && e <= 9311 || e >= 9472 && e <= 9654 || e === 9655 || e >= 9656 && e <= 9664 || e === 9665 || e >= 9666 && e <= 9719 || e >= 9720 && e <= 9727 || e >= 9728 && e <= 9838 || e === 9839 || e >= 9840 && e <= 10087 || e === 10088 || e === 10089 || e === 10090 || e === 10091 || e === 10092 || e === 10093 || e === 10094 || e === 10095 || e === 10096 || e === 10097 || e === 10098 || e === 10099 || e === 10100 || e === 10101 || e >= 10132 && e <= 10175 || e >= 10176 && e <= 10180 || e === 10181 || e === 10182 || e >= 10183 && e <= 10213 || e === 10214 || e === 10215 || e === 10216 || e === 10217 || e === 10218 || e === 10219 || e === 10220 || e === 10221 || e === 10222 || e === 10223 || e >= 10224 && e <= 10239 || e >= 10240 && e <= 10495 || e >= 10496 && e <= 10626 || e === 10627 || e === 10628 || e === 10629 || e === 10630 || e === 10631 || e === 10632 || e === 10633 || e === 10634 || e === 10635 || e === 10636 || e === 10637 || e === 10638 || e === 10639 || e === 10640 || e === 10641 || e === 10642 || e === 10643 || e === 10644 || e === 10645 || e === 10646 || e === 10647 || e === 10648 || e >= 10649 && e <= 10711 || e === 10712 || e === 10713 || e === 10714 || e === 10715 || e >= 10716 && e <= 10747 || e === 10748 || e === 10749 || e >= 10750 && e <= 11007 || e >= 11008 && e <= 11055 || e >= 11056 && e <= 11076 || e >= 11077 && e <= 11078 || e >= 11079 && e <= 11084 || e >= 11085 && e <= 11123 || e >= 11124 && e <= 11125 || e >= 11126 && e <= 11157 || e === 11158 || e >= 11159 && e <= 11263 || e >= 11776 && e <= 11777 || e === 11778 || e === 11779 || e === 11780 || e === 11781 || e >= 11782 && e <= 11784 || e === 11785 || e === 11786 || e === 11787 || e === 11788 || e === 11789 || e >= 11790 && e <= 11798 || e === 11799 || e >= 11800 && e <= 11801 || e === 11802 || e === 11803 || e === 11804 || e === 11805 || e >= 11806 && e <= 11807 || e === 11808 || e === 11809 || e === 11810 || e === 11811 || e === 11812 || e === 11813 || e === 11814 || e === 11815 || e === 11816 || e === 11817 || e >= 11818 && e <= 11822 || e === 11823 || e >= 11824 && e <= 11833 || e >= 11834 && e <= 11835 || e >= 11836 && e <= 11839 || e === 11840 || e === 11841 || e === 11842 || e >= 11843 && e <= 11855 || e >= 11856 && e <= 11857 || e === 11858 || e >= 11859 && e <= 11903 || e >= 12289 && e <= 12291 || e === 12296 || e === 12297 || e === 12298 || e === 12299 || e === 12300 || e === 12301 || e === 12302 || e === 12303 || e === 12304 || e === 12305 || e >= 12306 && e <= 12307 || e === 12308 || e === 12309 || e === 12310 || e === 12311 || e === 12312 || e === 12313 || e === 12314 || e === 12315 || e === 12316 || e === 12317 || e >= 12318 && e <= 12319 || e === 12320 || e === 12336 || e === 64830 || e === 64831 || e >= 65093 && e <= 65094;
}
function Dr(e) {
  e.forEach(function(t) {
    if (delete t.location, ai(t) || si(t))
      for (var r in t.options)
        delete t.options[r].location, Dr(t.options[r].value);
    else ri(t) && li(t.style) || (ni(t) || ii(t)) && Or(t.style) ? delete t.style.location : oi(t) && Dr(t.children);
  });
}
function Js(e, t) {
  t === void 0 && (t = {}), t = k({ shouldParseSkeletons: !0, requiresOtherClause: !0 }, t);
  var r = new Ws(e, t).parse();
  if (r.err) {
    var n = SyntaxError(L[r.err.kind]);
    throw n.location = r.err.location, n.originalMessage = r.err.message, n;
  }
  return t?.captureLocation || Dr(r.val), r.val;
}
var ht;
(function(e) {
  e.MISSING_VALUE = "MISSING_VALUE", e.INVALID_VALUE = "INVALID_VALUE", e.MISSING_INTL_API = "MISSING_INTL_API";
})(ht || (ht = {}));
var tr = (
  /** @class */
  (function(e) {
    er(t, e);
    function t(r, n, i) {
      var a = e.call(this, r) || this;
      return a.code = n, a.originalMessage = i, a;
    }
    return t.prototype.toString = function() {
      return "[formatjs Error: ".concat(this.code, "] ").concat(this.message);
    }, t;
  })(Error)
), Hn = (
  /** @class */
  (function(e) {
    er(t, e);
    function t(r, n, i, a) {
      return e.call(this, 'Invalid values for "'.concat(r, '": "').concat(n, '". Options are "').concat(Object.keys(i).join('", "'), '"'), ht.INVALID_VALUE, a) || this;
    }
    return t;
  })(tr)
), Qs = (
  /** @class */
  (function(e) {
    er(t, e);
    function t(r, n, i) {
      return e.call(this, 'Value for "'.concat(r, '" must be of type ').concat(n), ht.INVALID_VALUE, i) || this;
    }
    return t;
  })(tr)
), Ks = (
  /** @class */
  (function(e) {
    er(t, e);
    function t(r, n) {
      return e.call(this, 'The intl string context variable "'.concat(r, '" was not provided to the string "').concat(n, '"'), ht.MISSING_VALUE, n) || this;
    }
    return t;
  })(tr)
), ve;
(function(e) {
  e[e.literal = 0] = "literal", e[e.object = 1] = "object";
})(ve || (ve = {}));
function $s(e) {
  return e.length < 2 ? e : e.reduce(function(t, r) {
    var n = t[t.length - 1];
    return !n || n.type !== ve.literal || r.type !== ve.literal ? t.push(r) : n.value += r.value, t;
  }, []);
}
function eo(e) {
  return typeof e == "function";
}
function Ft(e, t, r, n, i, a, o) {
  if (e.length === 1 && yn(e[0]))
    return [
      {
        type: ve.literal,
        value: e[0].value
      }
    ];
  for (var s = [], u = 0, l = e; u < l.length; u++) {
    var h = l[u];
    if (yn(h)) {
      s.push({
        type: ve.literal,
        value: h.value
      });
      continue;
    }
    if (Es(h)) {
      typeof a == "number" && s.push({
        type: ve.literal,
        value: r.getNumberFormat(t).format(a)
      });
      continue;
    }
    var c = h.value;
    if (!(i && c in i))
      throw new Ks(c, o);
    var m = i[c];
    if (xs(h)) {
      (!m || typeof m == "string" || typeof m == "number") && (m = typeof m == "string" || typeof m == "number" ? String(m) : ""), s.push({
        type: typeof m == "string" ? ve.literal : ve.object,
        value: m
      });
      continue;
    }
    if (ni(h)) {
      var T = typeof h.style == "string" ? n.date[h.style] : Or(h.style) ? h.style.parsedOptions : void 0;
      s.push({
        type: ve.literal,
        value: r.getDateTimeFormat(t, T).format(m)
      });
      continue;
    }
    if (ii(h)) {
      var T = typeof h.style == "string" ? n.time[h.style] : Or(h.style) ? h.style.parsedOptions : n.time.medium;
      s.push({
        type: ve.literal,
        value: r.getDateTimeFormat(t, T).format(m)
      });
      continue;
    }
    if (ri(h)) {
      var T = typeof h.style == "string" ? n.number[h.style] : li(h.style) ? h.style.parsedOptions : void 0;
      T && T.scale && (m = m * (T.scale || 1)), s.push({
        type: ve.literal,
        value: r.getNumberFormat(t, T).format(m)
      });
      continue;
    }
    if (oi(h)) {
      var d = h.children, _ = h.value, P = i[_];
      if (!eo(P))
        throw new Qs(_, "function", o);
      var p = Ft(d, t, r, n, i, a), b = P(p.map(function(x) {
        return x.value;
      }));
      Array.isArray(b) || (b = [b]), s.push.apply(s, b.map(function(x) {
        return {
          type: typeof x == "string" ? ve.literal : ve.object,
          value: x
        };
      }));
    }
    if (ai(h)) {
      var w = h.options[m] || h.options.other;
      if (!w)
        throw new Hn(h.value, m, Object.keys(h.options), o);
      s.push.apply(s, Ft(w.value, t, r, n, i));
      continue;
    }
    if (si(h)) {
      var w = h.options["=".concat(m)];
      if (!w) {
        if (!Intl.PluralRules)
          throw new tr(`Intl.PluralRules is not available in this environment.
Try polyfilling it using "@formatjs/intl-pluralrules"
`, ht.MISSING_INTL_API, o);
        var E = r.getPluralRules(t, { type: h.pluralType }).select(m - (h.offset || 0));
        w = h.options[E] || h.options.other;
      }
      if (!w)
        throw new Hn(h.value, m, Object.keys(h.options), o);
      s.push.apply(s, Ft(w.value, t, r, n, i, m - (h.offset || 0)));
      continue;
    }
  }
  return $s(s);
}
function to(e, t) {
  return t ? k(k(k({}, e || {}), t || {}), Object.keys(e).reduce(function(r, n) {
    return r[n] = k(k({}, e[n]), t[n] || {}), r;
  }, {})) : e;
}
function ro(e, t) {
  return t ? Object.keys(e).reduce(function(r, n) {
    return r[n] = to(e[n], t[n]), r;
  }, k({}, e)) : e;
}
function _r(e) {
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
function no(e) {
  return e === void 0 && (e = {
    number: {},
    dateTime: {},
    pluralRules: {}
  }), {
    getNumberFormat: mr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.NumberFormat).bind.apply(t, vr([void 0], r, !1)))();
    }, {
      cache: _r(e.number),
      strategy: gr.variadic
    }),
    getDateTimeFormat: mr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.DateTimeFormat).bind.apply(t, vr([void 0], r, !1)))();
    }, {
      cache: _r(e.dateTime),
      strategy: gr.variadic
    }),
    getPluralRules: mr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.PluralRules).bind.apply(t, vr([void 0], r, !1)))();
    }, {
      cache: _r(e.pluralRules),
      strategy: gr.variadic
    })
  };
}
var io = (
  /** @class */
  (function() {
    function e(t, r, n, i) {
      r === void 0 && (r = e.defaultLocale);
      var a = this;
      if (this.formatterCache = {
        number: {},
        dateTime: {},
        pluralRules: {}
      }, this.format = function(u) {
        var l = a.formatToParts(u);
        if (l.length === 1)
          return l[0].value;
        var h = l.reduce(function(c, m) {
          return !c.length || m.type !== ve.literal || typeof c[c.length - 1] != "string" ? c.push(m.value) : c[c.length - 1] += m.value, c;
        }, []);
        return h.length <= 1 ? h[0] || "" : h;
      }, this.formatToParts = function(u) {
        return Ft(a.ast, a.locales, a.formatters, a.formats, u, void 0, a.message);
      }, this.resolvedOptions = function() {
        var u;
        return {
          locale: ((u = a.resolvedLocale) === null || u === void 0 ? void 0 : u.toString()) || Intl.NumberFormat.supportedLocalesOf(a.locales)[0]
        };
      }, this.getAst = function() {
        return a.ast;
      }, this.locales = r, this.resolvedLocale = e.resolveLocale(r), typeof t == "string") {
        if (this.message = t, !e.__parse)
          throw new TypeError("IntlMessageFormat.__parse must be set to process `message` of type `string`");
        var o = i || {};
        o.formatters;
        var s = ds(o, ["formatters"]);
        this.ast = e.__parse(t, k(k({}, s), { locale: this.resolvedLocale }));
      } else
        this.ast = t;
      if (!Array.isArray(this.ast))
        throw new TypeError("A message must be provided as a String or AST.");
      this.formats = ro(e.formats, n), this.formatters = i && i.formatters || no(this.formatterCache);
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
    }, e.__parse = Js, e.formats = {
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
function ao(e, t) {
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
const Xe = {}, so = (e, t, r) => r && (t in Xe || (Xe[t] = {}), e in Xe[t] || (Xe[t][e] = r), r), mi = (e, t) => {
  if (t == null)
    return;
  if (t in Xe && e in Xe[t])
    return Xe[t][e];
  const r = rr(t);
  for (let n = 0; n < r.length; n++) {
    const i = r[n], a = lo(i, e);
    if (a)
      return so(e, t, a);
  }
};
let qr;
const Bt = It({});
function oo(e) {
  return qr[e] || null;
}
function gi(e) {
  return e in qr;
}
function lo(e, t) {
  if (!gi(e))
    return null;
  const r = oo(e);
  return ao(r, t);
}
function uo(e) {
  if (e == null)
    return;
  const t = rr(e);
  for (let r = 0; r < t.length; r++) {
    const n = t[r];
    if (gi(n))
      return n;
  }
}
function fo(e, ...t) {
  delete Xe[e], Bt.update((r) => (r[e] = cs.all([r[e] || {}, ...t]), r));
}
dt(
  [Bt],
  ([e]) => Object.keys(e)
);
Bt.subscribe((e) => qr = e);
const Gt = {};
function ho(e, t) {
  Gt[e].delete(t), Gt[e].size === 0 && delete Gt[e];
}
function bi(e) {
  return Gt[e];
}
function co(e) {
  return rr(e).map((t) => {
    const r = bi(t);
    return [t, r ? [...r] : []];
  }).filter(([, t]) => t.length > 0);
}
function kr(e) {
  return e == null ? !1 : rr(e).some(
    (t) => {
      var r;
      return (r = bi(t)) == null ? void 0 : r.size;
    }
  );
}
function po(e, t) {
  return Promise.all(
    t.map((n) => (ho(e, n), n().then((i) => i.default || i)))
  ).then((n) => fo(e, ...n));
}
const xt = {};
function _i(e) {
  if (!kr(e))
    return e in xt ? xt[e] : Promise.resolve();
  const t = co(e);
  return xt[e] = Promise.all(
    t.map(
      ([r, n]) => po(r, n)
    )
  ).then(() => {
    if (kr(e))
      return _i(e);
    delete xt[e];
  }), xt[e];
}
const vo = {
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
  formats: vo,
  warnOnMissingMessages: !0,
  handleMissingMessage: void 0,
  ignoreTag: !0
}, go = mo;
function ct() {
  return go;
}
const yr = It(!1);
var bo = Object.defineProperty, _o = Object.defineProperties, yo = Object.getOwnPropertyDescriptors, In = Object.getOwnPropertySymbols, xo = Object.prototype.hasOwnProperty, Eo = Object.prototype.propertyIsEnumerable, Bn = (e, t, r) => t in e ? bo(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, wo = (e, t) => {
  for (var r in t || (t = {}))
    xo.call(t, r) && Bn(e, r, t[r]);
  if (In)
    for (var r of In(t))
      Eo.call(t, r) && Bn(e, r, t[r]);
  return e;
}, To = (e, t) => _o(e, yo(t));
let Ur;
const Wt = It(null);
function Pn(e) {
  return e.split("-").map((t, r, n) => n.slice(0, r + 1).join("-")).reverse();
}
function rr(e, t = ct().fallbackLocale) {
  const r = Pn(e);
  return t ? [.../* @__PURE__ */ new Set([...r, ...Pn(t)])] : r;
}
function rt() {
  return Ur ?? void 0;
}
Wt.subscribe((e) => {
  Ur = e ?? void 0, typeof window < "u" && e != null && document.documentElement.setAttribute("lang", e);
});
const So = (e) => {
  if (e && uo(e) && kr(e)) {
    const { loadingDelay: t } = ct();
    let r;
    return typeof window < "u" && rt() != null && t ? r = window.setTimeout(
      () => yr.set(!0),
      t
    ) : yr.set(!0), _i(e).then(() => {
      Wt.set(e);
    }).finally(() => {
      clearTimeout(r), yr.set(!1);
    });
  }
  return Wt.set(e);
}, pt = To(wo({}, Wt), {
  set: So
}), nr = (e) => {
  const t = /* @__PURE__ */ Object.create(null);
  return (n) => {
    const i = JSON.stringify(n);
    return i in t ? t[i] : t[i] = e(n);
  };
};
var Ao = Object.defineProperty, Zt = Object.getOwnPropertySymbols, yi = Object.prototype.hasOwnProperty, xi = Object.prototype.propertyIsEnumerable, Mn = (e, t, r) => t in e ? Ao(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, Yr = (e, t) => {
  for (var r in t || (t = {}))
    yi.call(t, r) && Mn(e, r, t[r]);
  if (Zt)
    for (var r of Zt(t))
      xi.call(t, r) && Mn(e, r, t[r]);
  return e;
}, vt = (e, t) => {
  var r = {};
  for (var n in e)
    yi.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && Zt)
    for (var n of Zt(e))
      t.indexOf(n) < 0 && xi.call(e, n) && (r[n] = e[n]);
  return r;
};
const St = (e, t) => {
  const { formats: r } = ct();
  if (e in r && t in r[e])
    return r[e][t];
  throw new Error(`[svelte-i18n] Unknown "${t}" ${e} format.`);
}, Ho = nr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = vt(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format numbers');
    return n && (i = St("number", n)), new Intl.NumberFormat(r, i);
  }
), Io = nr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = vt(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format dates');
    return n ? i = St("date", n) : Object.keys(i).length === 0 && (i = St("date", "short")), new Intl.DateTimeFormat(r, i);
  }
), Bo = nr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = vt(t, ["locale", "format"]);
    if (r == null)
      throw new Error(
        '[svelte-i18n] A "locale" must be set to format time values'
      );
    return n ? i = St("time", n) : Object.keys(i).length === 0 && (i = St("time", "short")), new Intl.DateTimeFormat(r, i);
  }
), Po = (e = {}) => {
  var t = e, {
    locale: r = rt()
  } = t, n = vt(t, [
    "locale"
  ]);
  return Ho(Yr({ locale: r }, n));
}, Mo = (e = {}) => {
  var t = e, {
    locale: r = rt()
  } = t, n = vt(t, [
    "locale"
  ]);
  return Io(Yr({ locale: r }, n));
}, Oo = (e = {}) => {
  var t = e, {
    locale: r = rt()
  } = t, n = vt(t, [
    "locale"
  ]);
  return Bo(Yr({ locale: r }, n));
}, Co = nr(
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  (e, t = rt()) => new io(e, t, ct().formats, {
    ignoreTag: ct().ignoreTag
  })
), Lo = (e, t = {}) => {
  var r, n, i, a;
  let o = t;
  typeof e == "object" && (o = e, e = o.id);
  const {
    values: s,
    locale: u = rt(),
    default: l
  } = o;
  if (u == null)
    throw new Error(
      "[svelte-i18n] Cannot format a message without first setting the initial locale."
    );
  let h = mi(e, u);
  if (!h)
    h = (a = (i = (n = (r = ct()).handleMissingMessage) == null ? void 0 : n.call(r, { locale: u, id: e, defaultValue: l })) != null ? i : l) != null ? a : e;
  else if (typeof h != "string")
    return console.warn(
      `[svelte-i18n] Message with id "${e}" must be of type "string", found: "${typeof h}". Gettin its value through the "$format" method is deprecated; use the "json" method instead.`
    ), h;
  if (!s)
    return h;
  let c = h;
  try {
    c = Co(h, u).format(s);
  } catch (m) {
    m instanceof Error && console.warn(
      `[svelte-i18n] Message "${e}" has syntax error:`,
      m.message
    );
  }
  return c;
}, No = (e, t) => Oo(t).format(e), Ro = (e, t) => Mo(t).format(e), Do = (e, t) => Po(t).format(e), ko = (e, t = rt()) => mi(e, t);
dt([pt, Bt], () => Lo);
dt([pt], () => No);
dt([pt], () => Ro);
dt([pt], () => Do);
dt([pt, Bt], () => ko);
const Uo = "__i18n__", Fo = [
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
function jo(e) {
  return typeof e == "string" && e.includes(Uo);
}
class Vo {
  load_component;
  #t = le(zt({}));
  get shared() {
    return g(this.#t);
  }
  set shared(t) {
    D(this.#t, t, !0);
  }
  #r = le(zt({}));
  get props() {
    return g(this.#r);
  }
  set props(t) {
    D(this.#r, t, !0);
  }
  #e = le((t) => t);
  get i18n() {
    return g(this.#e);
  }
  set i18n(t) {
    D(this.#e, t, !0);
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
    for (const n of Fo)
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
    ), Ee(() => {
      for (const n in t.shared_props)
        this._is_i18n_managed(`shared.${n}`, t.shared_props[n]) || (this.shared[n] = t.shared_props[n]);
      for (const n in t.props)
        this._is_i18n_managed(`props.${n}`, t.props[n]) || (this.props[n] = t.props[n]);
      this.register_component(
        t.shared_props.id,
        // @ts-ignore
        this.set_data.bind(this),
        this.get_data.bind(this)
      ), ne(() => {
        this.shared.id = t.shared_props.id;
      });
    }), Object.keys(this.translatable_props).length > 0 && pt.subscribe(() => {
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
    return _a(this.props);
  }
  update(t) {
    this.set_data(t);
  }
  set_data(t) {
    for (const r in t) {
      const n = t[r], i = jo(n) ? this._translate_and_store(this.shared_props.includes(r) ? "shared" : "props", r, n) : n;
      if (this.shared_props.includes(r)) {
        const a = r;
        this.shared[a] = i;
        continue;
      }
      this.props[r] = i;
    }
  }
  watch_for_change() {
    Ee(() => {
      this.mounted || (this.old_value = this.props.value, this.mounted = !0), this.old_value != this.props.value && (this.old_value = this.props.value, this.dispatch("change"));
    });
  }
}
Ht(["click"]);
function xr(e) {
  let t = ["", "k", "M", "G", "T", "P", "E", "Z"], r = 0;
  for (; e > 1e3 && r < t.length - 1; )
    e /= 1e3, r++;
  let n = t[r];
  return (Number.isInteger(e) ? e : e.toFixed(1)) + n;
}
function On(e) {
  return Object.prototype.toString.call(e) === "[object Date]";
}
function Fr(e, t, r, n) {
  if (typeof r == "number" || On(r)) {
    const i = n - r, a = (r - t) / (e.dt || 1 / 60), o = e.opts.stiffness * i, s = e.opts.damping * a, u = (o - s) * e.inv_mass, l = (a + u) * e.dt;
    return Math.abs(l) < e.opts.precision && Math.abs(i) < e.opts.precision ? n : (e.settled = !1, On(r) ? new Date(r.getTime() + l) : r + l);
  } else {
    if (Array.isArray(r))
      return r.map(
        (i, a) => (
          // @ts-ignore
          Fr(e, t[a], r[a], n[a])
        )
      );
    if (typeof r == "object") {
      const i = {};
      for (const a in r)
        i[a] = Fr(e, t[a], r[a], n[a]);
      return i;
    } else
      throw new Error(`Cannot spring ${typeof r} values`);
  }
}
function Cn(e, t = {}) {
  const r = It(e), { stiffness: n = 0.15, damping: i = 0.8, precision: a = 0.01 } = t;
  let o, s, u, l = (
    /** @type {T} */
    e
  ), h = (
    /** @type {T | undefined} */
    e
  ), c = 1, m = 0, T = !1;
  function d(P, p = {}) {
    h = P;
    const b = u = {};
    return e == null || p.hard || _.stiffness >= 1 && _.damping >= 1 ? (T = !0, o = Te.now(), l = P, r.set(e = h), Promise.resolve()) : (p.soft && (m = 1 / ((p.soft === !0 ? 0.5 : +p.soft) * 60), c = 0), s || (o = Te.now(), T = !1, s = Ra((w) => {
      if (T)
        return T = !1, s = null, !1;
      c = Math.min(c + m, 1);
      const E = Math.min(w - o, 1e3 / 30), x = {
        inv_mass: c,
        opts: _,
        settled: !0,
        dt: E * 60 / 1e3
      }, A = Fr(x, l, e, h);
      return o = w, l = /** @type {T} */
      e, r.set(e = /** @type {T} */
      A), x.settled && (s = null), !x.settled;
    })), new Promise((w) => {
      s.promise.then(() => {
        b === u && w();
      });
    }));
  }
  const _ = {
    set: d,
    update: (P, p) => d(P(
      /** @type {T} */
      h,
      /** @type {T} */
      e
    ), p),
    subscribe: r.subscribe,
    stiffness: n,
    damping: i,
    precision: a
  };
  return _;
}
var zo = /* @__PURE__ */ fe('<div><svg viewBox="-1200 -1200 3000 3000" fill="none" xmlns="http://www.w3.org/2000/svg" class="svelte-13ob5rg"><g><path d="M255.926 0.754768L509.702 139.936V221.027L255.926 81.8465V0.754768Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-13ob5rg"></path><path d="M509.69 139.936L254.981 279.641V361.255L509.69 221.55V139.936Z" fill="#FF7C00" class="svelte-13ob5rg"></path><path d="M0.250138 139.937L254.981 279.641V361.255L0.250138 221.55V139.937Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-13ob5rg"></path><path d="M255.923 0.232622L0.236328 139.936V221.55L255.923 81.8469V0.232622Z" fill="#FF7C00" class="svelte-13ob5rg"></path></g><g><path d="M255.926 141.5L509.702 280.681V361.773L255.926 222.592V141.5Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-13ob5rg"></path><path d="M509.69 280.679L254.981 420.384V501.998L509.69 362.293V280.679Z" fill="#FF7C00" class="svelte-13ob5rg"></path><path d="M0.250138 280.681L254.981 420.386V502L0.250138 362.295V280.681Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-13ob5rg"></path><path d="M255.923 140.977L0.236328 280.68V362.294L255.923 222.591V140.977Z" fill="#FF7C00" class="svelte-13ob5rg"></path></g></svg></div>');
function Xo(e, t) {
  Kt(t, !0);
  const r = () => sn(u, "$top", i), n = () => sn(l, "$bottom", i), [i, a] = Ea();
  var o = this && this.__awaiter || function(w, E, x, A) {
    function H(B) {
      return B instanceof x ? B : new x(function(O) {
        O(B);
      });
    }
    return new (x || (x = Promise))(function(B, O) {
      function U(z) {
        try {
          he(A.next(z));
        } catch (G) {
          O(G);
        }
      }
      function $(z) {
        try {
          he(A.throw(z));
        } catch (G) {
          O(G);
        }
      }
      function he(z) {
        z.done ? B(z.value) : H(z.value).then(U, $);
      }
      he((A = A.apply(w, E || [])).next());
    });
  };
  let s = I(t, "margin", 3, !0);
  const u = Cn([0, 0]), l = Cn([0, 0]);
  let h = le(!1);
  function c() {
    return o(this, void 0, void 0, function* () {
      yield Promise.all([u.set([125, 140]), l.set([-125, -140])]), yield Promise.all([u.set([-125, 140]), l.set([125, -140])]), yield Promise.all([u.set([-125, 0]), l.set([125, -0])]), yield Promise.all([u.set([125, 0]), l.set([-125, 0])]);
    });
  }
  function m() {
    return o(this, void 0, void 0, function* () {
      yield c(), g(h) || m();
    });
  }
  function T() {
    return o(this, void 0, void 0, function* () {
      yield Promise.all([u.set([125, 0]), l.set([-125, 0])]), m();
    });
  }
  Ee(() => (T(), () => {
    D(h, !0);
  }));
  var d = zo();
  let _;
  var P = se(d), p = se(P), b = Z(p);
  K(() => {
    _ = $e(d, 1, "svelte-13ob5rg", null, _, { margin: s() }), Ce(p, `transform: translate(${r()[0] ?? ""}px, ${r()[1] ?? ""}px);`), Ce(b, `transform: translate(${n()[0] ?? ""}px, ${n()[1] ?? ""}px);`);
  }), N(e, d), Qt(), a();
}
var Wo = function(e, t, r, n) {
  function i(a) {
    return a instanceof r ? a : new r(function(o) {
      o(a);
    });
  }
  return new (r || (r = Promise))(function(a, o) {
    function s(h) {
      try {
        l(n.next(h));
      } catch (c) {
        o(c);
      }
    }
    function u(h) {
      try {
        l(n.throw(h));
      } catch (c) {
        o(c);
      }
    }
    function l(h) {
      h.done ? a(h.value) : i(h.value).then(s, u);
    }
    l((n = n.apply(e, t || [])).next());
  });
};
let kt = [], Er = !1;
const Zo = typeof window < "u", Ei = Zo ? window.requestAnimationFrame : (e) => {
};
function qo(e) {
  return Wo(this, arguments, void 0, function* (t, r = !0) {
    if (!(window.__gradio_mode__ === "website" || window.__gradio_mode__ !== "app" && r !== !0)) {
      if (kt.push(t), !Er) Er = !0;
      else return;
      yield ga(), Ei(() => {
        let n = [0, 0];
        for (let i = 0; i < kt.length; i++) {
          const o = kt[i].getBoundingClientRect();
          (i === 0 || o.top + window.scrollY <= n[0]) && (n[0] = o.top + window.scrollY, n[1] = i);
        }
        window.scrollTo({ top: n[0] - 20, behavior: "smooth" }), Er = !1, kt = [];
      });
    }
  });
}
var Yo = /* @__PURE__ */ fe('<div class="validation-error svelte-1aeuuvr"> <button class="svelte-1aeuuvr"><!></button></div>'), Jo = /* @__PURE__ */ fe('<div class="eta-bar svelte-1aeuuvr"></div>'), Qo = /* @__PURE__ */ fe("<!> ", 1), Ko = /* @__PURE__ */ fe("<!> <!> <!> <!>", 1), $o = /* @__PURE__ */ fe('<div class="progress-level svelte-1aeuuvr"><div class="progress-level-inner svelte-1aeuuvr"><!></div> <div class="progress-bar-wrap svelte-1aeuuvr"><div class="progress-bar svelte-1aeuuvr"></div></div></div>'), el = /* @__PURE__ */ fe('<p class="loading svelte-1aeuuvr"> </p> <!>', 1), tl = /* @__PURE__ */ fe("<!> <div><!> <!></div> <!> <!>", 1), rl = /* @__PURE__ */ fe('<div class="clear-status svelte-1aeuuvr"><!></div> <span class="error svelte-1aeuuvr"> </span> <!>', 1), nl = /* @__PURE__ */ fe("<div> <!> </div>"), il = /* @__PURE__ */ fe('<div data-testid="status-tracker"><!> <!></div> <!>', 1);
function al(e, t) {
  Kt(t, !0);
  let r = I(t, "eta", 3, null), n = I(t, "scroll_to_output", 3, !1), i = I(t, "timer", 3, !0), a = I(t, "show_progress", 3, "full"), o = I(t, "message", 3, null), s = I(t, "progress", 3, null), u = I(t, "variant", 3, "default"), l = I(t, "loading_text", 3, "Loading..."), h = I(t, "absolute", 3, !0), c = I(t, "translucent", 3, !1), m = I(t, "border", 3, !1), T = I(t, "validation_error", 7, null), d = I(t, "show_validation_error", 3, !0), _ = I(t, "type", 3, null), P = I(t, "used_cache", 3, null), p = I(t, "cache_duration", 3, null), b = I(t, "avg_time", 3, null), w, E = !1, x = le(0), A = le(null), H = le(null), B = le(!1), O = le(null), U = le(!1), $ = le(!1), he = le(null), z = le(null), G = le("from cache"), Ue = le(!1), be = null, Se = null;
  const Ae = Oe(() => !(d() && T()) && (_() === "input" || !t.status || t.status === "complete" || a() === "hidden" || t.status == "streaming"));
  let Fe = le(0);
  const ce = Oe(() => g(H) === null || g(H) <= 0 || !g(Fe) ? 0 : Math.min(g(Fe) / g(H), 1)), ie = Oe(() => g(Fe).toFixed(1));
  let He = Oe(() => s() == null), _e = Oe(() => r() !== null && r() !== void 0 ? r() : g(A));
  function Le() {
    Ei(() => {
      D(Fe, (performance.now() - g(x)) / 1e3), E && Le();
    });
  }
  let oe = Oe(() => {
    let j = null;
    s() != null ? j = s().map((te) => {
      if (te.index != null && te.length != null)
        return te.index / te.length;
      if (te.progress != null)
        return te.progress;
    }) : j = null;
    let J, ee = "";
    return j ? (J = j[j.length - 1], J === 0 ? ee = "0" : ee = "150ms") : J = void 0, {
      progress_level: j,
      last_progress_level: J,
      progress_bar_transition: ee
    };
  });
  function ue() {
    E || (D(A, D(O, null), !0), D(x, performance.now(), !0), E = !0, Le());
  }
  function Ie() {
    D(A, D(O, null), !0), E && (E = !1);
  }
  Ee(() => {
    t.status === "pending" ? ue() : ne(() => {
      Ie();
    });
  }), Ee(() => {
    w && n() && (t.status === "pending" || t.status === "complete") && qo(w, t.autoscroll);
  }), Ee(() => {
    g(_e) != null && g(A) !== g(_e) && (D(H, (performance.now() - g(x)) / 1e3 + g(_e)), D(O, g(H).toFixed(1), !0), D(A, g(_e), !0));
  });
  function Ne() {
    D(B, !1);
  }
  Ee(() => {
    ne(() => {
      Ne();
    }), t.status === "error" && o() && D(B, !0);
  }), Ee(() => {
    t.status === "complete" && _() === "output" && P() && p() != null && (D(he, p().toFixed(1), !0), D(G, P() === "full" ? "from cache" : "used cache", !0), D(Ue, b() != null && b() > p() && b() > 0, !0), D(z, g(Ue) ? b().toFixed(1) : null, !0), D(U, !0), D($, !1), be && clearTimeout(be), Se && clearTimeout(Se), be = setTimeout(
      () => {
        D($, !0), Se = setTimeout(
          () => {
            D(U, !1), D($, !1);
          },
          500
        );
      },
      1750
    ));
  });
  var Ge = il(), Be = pe(Ge);
  let Ze, ae;
  var Re = se(Be);
  {
    var Pt = (j) => {
      var J = Yo(), ee = se(J), te = Z(ee), me = se(te);
      {
        let we = Oe(() => t.i18n ? t.i18n("common.clear") : "Clear");
        mn(me, {
          get Icon() {
            return gn;
          },
          get label() {
            return g(we);
          },
          disabled: !1,
          size: "x-small",
          background: "var(--background-fill-primary)",
          color: "var(--error-background-text)",
          border: "var(--border-color-primary)",
          onclick: () => T(null)
        });
      }
      K(() => ge(ee, `${T() ?? ""} `)), N(j, J);
    };
    Y(Re, (j) => {
      T() && d() && j(Pt);
    });
  }
  var Mt = Z(Re, 2);
  {
    var Ot = (j) => {
      var J = tl(), ee = pe(J);
      {
        var te = (F) => {
          var X = Jo();
          let f;
          K(() => f = Ce(X, "", f, {
            transform: `translateX(${(g(ce) || 0) * 100 - 100}%)`
          })), N(F, X);
        };
        Y(ee, (F) => {
          u() === "default" && g(He) && a() === "full" && F(te);
        });
      }
      var me = Z(ee, 2);
      let we;
      var je = se(me);
      {
        var re = (F) => {
          var X = st(), f = pe(X);
          un(f, 17, s, on, (v, y) => {
            var S = st(), M = pe(S);
            {
              var C = (W) => {
                var q = Qo(), Pe = pe(q);
                {
                  var Ye = (ye) => {
                    var Je = Me();
                    K((gt, bt) => ge(Je, `${gt ?? ""}/${bt ?? ""}`), [
                      () => xr(g(y).index || 0),
                      () => xr(g(y).length)
                    ]), N(ye, Je);
                  }, Q = (ye) => {
                    var Je = Me();
                    K((gt) => ge(Je, gt), [() => xr(g(y).index || 0)]), N(ye, Je);
                  };
                  Y(Pe, (ye) => {
                    g(y).length != null ? ye(Ye) : ye(Q, -1);
                  });
                }
                var Ve = Z(Pe);
                K(() => ge(Ve, ` ${g(y).unit ?? ""} |  `)), N(W, q);
              };
              Y(M, (W) => {
                g(y).index != null && W(C);
              });
            }
            N(v, S);
          }), N(F, X);
        }, qe = (F) => {
          var X = Me();
          K(() => ge(X, `queue: ${t.queue_position + 1}/${t.queue_size ?? ""} |`)), N(F, X);
        }, nt = (F) => {
          var X = Me("processing |");
          N(F, X);
        };
        Y(je, (F) => {
          s() ? F(re) : t.queue_position !== null && t.queue_size !== void 0 && t.queue_position >= 0 ? F(qe, 1) : t.queue_position === 0 && F(nt, 2);
        });
      }
      var or = Z(je, 2);
      {
        var lr = (F) => {
          var X = Me();
          K(() => ge(X, `${g(ie) ?? ""}${r() ? `/${g(O)}` : ""}s`)), N(F, X);
        };
        Y(or, (F) => {
          i() && F(lr);
        });
      }
      var Ct = Z(me, 2);
      {
        var Lt = (F) => {
          var X = $o(), f = se(X), v = se(f);
          {
            var y = (W) => {
              var q = st(), Pe = pe(q);
              un(Pe, 17, s, on, (Ye, Q, Ve) => {
                var ye = st(), Je = pe(ye);
                {
                  var gt = (bt) => {
                    var Jr = Ko(), Qr = pe(Jr);
                    {
                      var wi = (de) => {
                        var De = Me(" /");
                        N(de, De);
                      };
                      Y(Qr, (de) => {
                        Ve !== 0 && de(wi);
                      });
                    }
                    var Kr = Z(Qr, 2);
                    {
                      var Ti = (de) => {
                        var De = Me();
                        K(() => ge(De, g(Q).desc)), N(de, De);
                      };
                      Y(Kr, (de) => {
                        g(Q).desc != null && de(Ti);
                      });
                    }
                    var $r = Z(Kr, 2);
                    {
                      var Si = (de) => {
                        var De = Me("-");
                        N(de, De);
                      };
                      Y($r, (de) => {
                        g(Q).desc != null && g(oe).progress_level && g(oe).progress_level[Ve] != null && de(Si);
                      });
                    }
                    var Ai = Z($r, 2);
                    {
                      var Hi = (de) => {
                        var De = Me();
                        K((Ii) => ge(De, `${Ii ?? ""}%`), [
                          () => (100 * (g(oe).progress_level[Ve] || 0)).toFixed(1)
                        ]), N(de, De);
                      };
                      Y(Ai, (de) => {
                        g(oe).progress_level != null && de(Hi);
                      });
                    }
                    N(bt, Jr);
                  };
                  Y(Je, (bt) => {
                    (g(Q).desc != null || g(oe).progress_level && g(oe).progress_level[Ve] != null) && bt(gt);
                  });
                }
                N(Ye, ye);
              }), N(W, q);
            };
            Y(v, (W) => {
              s() != null && W(y);
            });
          }
          var S = Z(f, 2), M = se(S);
          let C;
          K(() => C = Ce(M, "", C, {
            width: `${g(oe).last_progress_level * 100}%`,
            transition: g(oe).progress_bar_transition
          })), N(F, X);
        }, ur = (F) => {
          {
            let X = Oe(() => u() === "default");
            Xo(F, {
              get margin() {
                return g(X);
              }
            });
          }
        };
        Y(Ct, (F) => {
          g(oe).last_progress_level != null ? F(Lt) : a() === "full" && F(ur, 1);
        });
      }
      var mt = Z(Ct, 2);
      {
        var Nt = (F) => {
          var X = el(), f = pe(X), v = se(f), y = Z(f, 2);
          Br(y, t, "additional-loading-text", {}), K(() => ge(v, l())), N(F, X);
        };
        Y(mt, (F) => {
          i() || F(Nt);
        });
      }
      K(() => we = $e(me, 1, "progress-text svelte-1aeuuvr", null, we, {
        "meta-text-center": u() === "center",
        "meta-text": u() === "default"
      })), N(j, J);
    }, ir = (j) => {
      var J = rl(), ee = pe(J), te = se(ee);
      {
        let re = Oe(() => t.i18n("common.clear"));
        mn(te, {
          get Icon() {
            return gn;
          },
          get label() {
            return g(re);
          },
          disabled: !1,
          $$events: {
            click: () => {
              t.on_clear_status?.();
            }
          }
        });
      }
      var me = Z(ee, 2), we = se(me), je = Z(me, 2);
      Br(je, t, "error", {}), K((re) => ge(we, re), [() => t.i18n("common.error")]), N(j, J);
    };
    Y(Mt, (j) => {
      t.status === "pending" ? j(Ot) : t.status === "error" && j(ir, 1);
    });
  }
  Xt(Be, (j) => w = j, () => w);
  var ar = Z(Be, 2);
  {
    var sr = (j) => {
      var J = nl();
      let ee, te;
      var me = se(J), we = Z(me);
      {
        var je = (qe) => {
          var nt = Me();
          K(() => ge(nt, `~${g(z) ?? ""}s
			→ `)), N(qe, nt);
        };
        Y(we, (qe) => {
          g(Ue) && qe(je);
        });
      }
      var re = Z(we);
      K(() => {
        ee = $e(J, 1, "cache-indicator svelte-1aeuuvr", null, ee, { "fade-out": g($) }), te = Ce(J, "", te, { position: h() ? "absolute" : "static" }), ge(me, `⚡ ${g(G) ?? ""}: `), ge(re, `${g(he) ?? ""}s`);
      }), N(j, J);
    };
    Y(ar, (j) => {
      g(U) && j(sr);
    });
  }
  K(() => {
    Ze = $e(Be, 1, `wrap ${u() ?? ""} ${a() ?? ""}`, "svelte-1aeuuvr", Ze, {
      "no-click": T() && d(),
      hide: g(Ae),
      translucent: u() === "center" && (t.status === "pending" || t.status === "error") || c() || a() === "minimal" || T(),
      generating: t.status === "generating" && a() === "full",
      border: m()
    }), ae = Ce(Be, "", ae, {
      position: h() ? "absolute" : "static",
      padding: h() ? "0" : "var(--size-8) 0"
    });
  }), N(e, Ge), Qt();
}
const sl = (e) => {
  const t = {};
  for (let r = 0, n = e.length; r < n; r++) {
    const i = e[r];
    for (const a in i)
      t[a] ? t[a] = t[a].concat(i[a]) : t[a] = i[a];
  }
  return t;
}, ol = [
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
], ll = [
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
], ul = [
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
sl([
  Object.fromEntries(ol.map((e) => [e, ["*"]])),
  Object.fromEntries(ll.map((e) => [e, ["svg:*"]])),
  Object.fromEntries(ul.map((e) => [e, ["math:*"]]))
]);
Ht(["touchstart", "touchmove", "touchend", "click", "keydown"]);
var fl = /* @__PURE__ */ new Set(["$$slots", "$$events", "$$legacy"]), hl = /* @__PURE__ */ fe('<!> <div class="repair-editor svelte-r41nsf"><div class="surface svelte-r41nsf" role="application" aria-label="Repair mask canvas"><canvas class="svelte-r41nsf"></canvas></div></div>', 1);
function dl(e, t) {
  Kt(t, !0);
  const r = /* @__PURE__ */ $a(t, fl), n = new Vo(r), i = "#f0445d", a = 0.45, o = 24, s = 1, u = 512;
  let l, h, c = le(zt({})), m = null, T = null, d = null, _ = null, P = 1, p = 1, b = 0, w = 0, E = "", x = "", A = 0, H = null, B = null, O = null, U = null, $ = !1, he = null, z = null, G = null;
  function Ue(f) {
    return JSON.parse(JSON.stringify(f || {}));
  }
  function be(f, v) {
    const y = Number(f);
    return Number.isFinite(y) ? y : v;
  }
  function Se(f, v, y) {
    return Math.max(v, Math.min(y, f));
  }
  function Ae(f, v) {
    const y = Math.round(be(f, v));
    return y <= 0 ? v : Math.max(1, y);
  }
  function Fe(f) {
    return typeof f == "number" ? String(f) + "px" : f || "520px";
  }
  function ce() {
    const f = g(c).tool;
    return f === "eraser" || f === "rect_add" || f === "rect_erase" ? f : "brush";
  }
  function ie() {
    return Se(be(g(c).preview_alpha, a), 0, 1);
  }
  function He() {
    return Se(be(g(c).brush_size, o), s, u);
  }
  function _e() {
    return typeof g(c).base_image == "string" && g(c).base_image ? g(c).base_image : null;
  }
  function Le() {
    for (const f of [g(c).mask_png])
      if (typeof f == "string" && f.startsWith("data:image/png;base64,")) return f;
    return null;
  }
  function oe(f) {
    const v = Ae(g(c)[f], 0);
    if (v > 0) return v;
    const y = f === "source_width" ? b : w;
    if (y > 0) return y;
    const S = m || T;
    return S ? f === "source_width" ? Math.max(1, S.naturalWidth) : Math.max(1, S.naturalHeight) : 1;
  }
  function ue() {
    return Ie(), {
      width: oe("source_width"),
      height: oe("source_height")
    };
  }
  function Ie() {
    const f = Ae(g(c).source_width, 0), v = Ae(g(c).source_height, 0);
    f > 0 && (b = f), v > 0 && (w = v);
    const y = m || T;
    y && (b <= 0 && (b = Math.max(1, y.naturalWidth)), w <= 0 && (w = Math.max(1, y.naturalHeight)));
  }
  function Ne(f) {
    return JSON.stringify([
      f.image_id || "",
      f.revision || 0,
      f.base_image || "",
      f.mask_png || "",
      f.source_width || 0,
      f.source_height || 0
    ]);
  }
  function Ge(f, v, y) {
    if (!f) {
      v === A && y(null);
      return;
    }
    const S = new Image();
    S.onload = () => {
      v === A && y(S);
    }, S.onerror = () => {
      v === A && y(null);
    }, S.src = f;
  }
  function Be(f, v) {
    const y = document.createElement("canvas");
    return y.width = f, y.height = v, y;
  }
  function Ze(f) {
    const v = ue(), y = Be(v.width, v.height), S = Be(v.width, v.height), M = y.getContext("2d", { willReadFrequently: !0 }), C = S.getContext("2d", { willReadFrequently: !0 });
    if (!M || !C) return;
    M.imageSmoothingEnabled = !1, C.imageSmoothingEnabled = !1, f && M.drawImage(f, 0, 0, v.width, v.height);
    const W = M.getImageData(0, 0, v.width, v.height), q = M.createImageData(v.width, v.height), Pe = C.createImageData(v.width, v.height), Ye = Re(i);
    for (let Q = 0; Q < W.data.length; Q += 4) {
      const Ve = Math.max(W.data[Q], W.data[Q + 1], W.data[Q + 2]), ye = Math.round(W.data[Q + 3] * Ve / 255);
      q.data[Q] = 255, q.data[Q + 1] = 255, q.data[Q + 2] = 255, q.data[Q + 3] = ye, Pe.data[Q] = Ye[0], Pe.data[Q + 1] = Ye[1], Pe.data[Q + 2] = Ye[2], Pe.data[Q + 3] = ye;
    }
    M.putImageData(q, 0, 0), C.putImageData(Pe, 0, 0), d = y, _ = S;
  }
  function ae() {
    const f = ue();
    d && _ && d.width === f.width && d.height === f.height && _.width === f.width && _.height === f.height || Ze(T);
  }
  function Re(f) {
    const v = f.match(/^#([0-9a-f]{6})$/i);
    return v ? [
      Number.parseInt(v[1].slice(0, 2), 16),
      Number.parseInt(v[1].slice(2, 4), 16),
      Number.parseInt(v[1].slice(4, 6), 16)
    ] : [240, 68, 93];
  }
  function Pt() {
    ae();
    const f = [];
    for (const v of [d, _]) {
      const y = v?.getContext("2d", { willReadFrequently: !0 });
      y && f.push(y);
    }
    return f;
  }
  function Mt(f, v) {
    const S = ce() === "eraser";
    for (const M of Pt())
      M.save(), M.globalCompositeOperation = S ? "destination-out" : "source-over", M.strokeStyle = M.canvas === d ? "#ffffff" : i, M.lineWidth = He(), M.lineCap = "round", M.lineJoin = "round", M.beginPath(), M.moveTo(f.x, f.y), M.lineTo(v.x, v.y), M.stroke(), M.restore();
  }
  function Ot(f, v) {
    const y = Math.min(f.x, v.x), S = Math.min(f.y, v.y);
    return {
      x: y,
      y: S,
      width: Math.max(1, Math.abs(v.x - f.x)),
      height: Math.max(1, Math.abs(v.y - f.y))
    };
  }
  function ir(f) {
    const v = ce() === "rect_erase";
    for (const y of Pt())
      y.save(), y.globalCompositeOperation = v ? "destination-out" : "source-over", y.fillStyle = y.canvas === d ? "#ffffff" : i, y.fillRect(f.x, f.y, f.width, f.height), y.restore();
  }
  function ar() {
    ae();
    const f = d?.getContext("2d", { willReadFrequently: !0 }), v = _?.getContext("2d", { willReadFrequently: !0 });
    return !f || !v || !d || !_ ? null : {
      mask: f.getImageData(0, 0, d.width, d.height),
      tint: v.getImageData(0, 0, _.width, _.height)
    };
  }
  function sr(f) {
    var v, y;
    !f || !d || !_ || ((v = d.getContext("2d")) === null || v === void 0 || v.putImageData(f.mask, 0, 0), (y = _.getContext("2d")) === null || y === void 0 || y.putImageData(f.tint, 0, 0));
  }
  function j() {
    const f = l?.getBoundingClientRect();
    return {
      width: Math.max(1, f?.width || 1),
      height: Math.max(1, f?.height || 1)
    };
  }
  function J(f, v) {
    const y = ue(), S = Math.min(f / y.width, v / y.height), M = y.width * S, C = y.height * S;
    return {
      left: (f - M) / 2,
      top: (v - C) / 2,
      width: M,
      height: C,
      scale: S
    };
  }
  function ee(f, v) {
    if (!l) return null;
    const y = l.getBoundingClientRect(), S = J(y.width, y.height), M = f.clientX - y.left, C = f.clientY - y.top;
    if (!(M >= S.left && M <= S.left + S.width && C >= S.top && C <= S.top + S.height) && !v) return null;
    const q = ue();
    return {
      x: Se((M - S.left) / S.scale, 0, q.width),
      y: Se((C - S.top) / S.scale, 0, q.height)
    };
  }
  function te() {
    if (!h) return;
    const f = j(), v = Math.max(1, Math.round(f.width)), y = Math.max(1, Math.round(f.height));
    (h.width !== v || h.height !== y) && (h.width = v, h.height = y), P = f.width, p = f.height;
  }
  function me() {
    G === null && (G = requestAnimationFrame(() => {
      G = null, te(), re();
    }));
  }
  function we(f, v, y) {
    f.fillStyle = "#f1f5f9", f.fillRect(0, 0, v, y), f.fillStyle = "#e2e8f0";
    const S = 16;
    for (let M = 0; M < y; M += S)
      for (let C = 0; C < v; C += S)
        (C / S + M / S) % 2 === 0 && f.fillRect(C, M, S, S);
  }
  function je(f, v, y) {
    const S = y.left + v.x * y.scale, M = y.top + v.y * y.scale, C = v.width * y.scale, W = v.height * y.scale;
    f.save(), f.fillStyle = ce() === "rect_erase" ? "rgba(37, 99, 235, 0.16)" : "rgba(240, 68, 93, 0.18)", f.strokeStyle = ce() === "rect_erase" ? "#2563eb" : i, f.lineWidth = 2, f.setLineDash([6, 4]), f.fillRect(S, M, C, W), f.strokeRect(S, M, C, W), f.restore();
  }
  function re() {
    if (!h) return;
    te(), ae();
    const f = h.getContext("2d");
    if (!f) return;
    const v = P, y = p;
    f.clearRect(0, 0, v, y), we(f, v, y);
    const S = J(v, y);
    m && (f.save(), f.imageSmoothingEnabled = !0, f.drawImage(m, S.left, S.top, S.width, S.height), f.restore()), _ && (f.save(), f.imageSmoothingEnabled = !1, f.globalAlpha = ie(), f.drawImage(_, S.left, S.top, S.width, S.height), f.restore()), he && je(f, he, S);
  }
  function qe(f) {
    ae();
    const v = ue(), y = Math.max(0, Math.trunc(be(g(c).revision, 0))) + 1, S = Object.assign(Object.assign({}, g(c)), {
      image_id: String(g(c).image_id || ""),
      revision: y,
      source_width: v.width,
      source_height: v.height,
      base_image: _e(),
      mask_png: d ? d.toDataURL("image/png") : null,
      preview_alpha: ie(),
      tool: ce(),
      brush_size: He(),
      status: f
    });
    D(c, S, !0), n.props.value = S, x = JSON.stringify(S), n.dispatch("change"), re();
  }
  function nt(f = "mask updated") {
    qe(f);
  }
  function or(f) {
    const v = Ue(f);
    D(c, v, !0);
    const y = Ne(v);
    if (y === E) {
      re();
      return;
    }
    E = y, A += 1, m = null, T = null, d = null, _ = null, b = Ae(v.source_width, 0), w = Ae(v.source_height, 0);
    const S = A, M = _e(), C = Le();
    Ge(M, S, (W) => {
      m = W, Ie(), ae(), re();
    }), Ge(C, S, (W) => {
      T = W, Ie(), Ze(T), re();
    }), !M && !C && (ae(), re());
  }
  function lr(f) {
    if (f.button !== 0 || H !== null) return;
    const v = ee(f, !1);
    v && (ae(), f.preventDefault(), H = f.pointerId, B = v, O = v, U = ar(), $ = !1, he = null, l.setPointerCapture(f.pointerId), (ce() === "brush" || ce() === "eraser") && (Mt(v, v), $ = !0, re()));
  }
  function Ct(f) {
    if (f.pointerId !== H || !B || !O) return;
    f.preventDefault();
    const v = ee(f, !0);
    v && (ce() === "brush" || ce() === "eraser" ? (Mt(O, v), $ = !0, O = v) : (he = Ot(B, v), $ = !0), re());
  }
  function Lt() {
    const f = H;
    H = null, f !== null && l?.hasPointerCapture(f) && l.releasePointerCapture(f), B = null, O = null, U = null, $ = !1, he = null;
  }
  function ur(f) {
    if (f.pointerId !== H || !B) return;
    f.preventDefault();
    const v = ee(f, !0) || O || B;
    (ce() === "rect_add" || ce() === "rect_erase") && ir(Ot(B, v));
    const y = $;
    Lt(), y ? nt() : re();
  }
  function mt() {
    H !== null && (sr(U), Lt(), re());
  }
  function Nt() {
    mt();
  }
  function F(f) {
    return z = new ResizeObserver(me), z.observe(f), me(), {
      destroy: () => {
        z?.disconnect(), z = null;
      }
    };
  }
  function X() {
    return "height:" + Fe(n.props.height);
  }
  Ee(() => {
    const f = JSON.stringify(n.props.value || null);
    f !== x && (x = f, or(n.props.value));
  }), Zn(() => (window.addEventListener("blur", Nt), te(), re(), () => window.removeEventListener("blur", Nt))), Ba(() => {
    A += 1, G !== null && cancelAnimationFrame(G), z?.disconnect();
  }), is(e, {
    get visible() {
      return n.shared.visible;
    },
    variant: "solid",
    border_mode: "base",
    padding: !1,
    get elem_id() {
      return n.shared.elem_id;
    },
    get elem_classes() {
      return n.shared.elem_classes;
    },
    allow_overflow: !1,
    get container() {
      return n.shared.container;
    },
    get scale() {
      return n.shared.scale;
    },
    get min_width() {
      return n.shared.min_width;
    },
    children: (f, v) => {
      var y = hl(), S = pe(y);
      al(S, ts(
        {
          get autoscroll() {
            return n.shared.autoscroll;
          },
          get i18n() {
            return n.i18n;
          }
        },
        () => n.shared.loading_status,
        {
          on_clear_status: () => n.dispatch("clear_status", n.shared.loading_status)
        }
      ));
      var M = Z(S, 2), C = se(M), W = se(C);
      Xt(W, (q) => h = q, () => h), Xt(C, (q) => l = q, () => l), ka(C, (q) => F?.(q)), K((q) => Ce(M, q), [() => X()]), Tt("pointerdown", C, lr), Tt("pointermove", C, Ct), Tt("pointerup", C, ur), Ar("pointercancel", C, mt), Ar("lostpointercapture", C, mt), N(f, y);
    },
    $$slots: { default: !0 }
  }), Qt();
}
Ht(["pointerdown", "pointermove", "pointerup"]);
export {
  dl as default
};

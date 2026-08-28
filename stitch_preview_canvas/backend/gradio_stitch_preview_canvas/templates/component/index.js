import { i as Yr, g as Un, o as Pi, n as et, u as ue, s as Ii, r as Mr, m as st, a as w, b as l, t as Jr, d as Bi, q as Oi, c as Fn, e as lt, f as ar, h as er, j as Mi, T as Li, k as Ni, l as tr, p as ot, v as Qr, w as ut, x as Gn, y as jn, z as Vn, A as Gt, E as sr, B as yt, C as zn, D as Pe, F as sn, G as Ci, H as Xn, I as Kr, J as Ri, K as on, L as Di, M as ki, N as Xe, O as qn, P as _r, Q as Ui, R as Fi, S as Gi, U as ji, V as Wn, W as $r, X as ln, Y as un, Z as Vi, _ as zi, $ as Xi, a0 as qi, a1 as Wi, a2 as Zi, a3 as Yi, a4 as Ji, a5 as en, a6 as Qi, a7 as Ke, a8 as jt, a9 as Ki, aa as $i, ab as ea, ac as ta, ad as ra, ae as na, af as tn, ag as ia, ah as aa, ai as He, aj as Lr, ak as Nr, al as sa, am as oa, an as Ut, ao as la, ap as ua, aq as fa, ar as ca, as as ha, at as Zn, au as Nt, av as q, aw as da, ax as fn, ay as pa, az as ge, aA as or, aB as lr, aC as J, aD as gt, aE as re, aF as ma, aG as oe, aH as ve, aI as Le, aJ as va } from "./render-DoYhCszp.js";
function Yn(e) {
  throw new Error("https://svelte.dev/e/lifecycle_outside_component");
}
const ga = [];
function ba(e, t = !1, r = !1) {
  return Qt(e, /* @__PURE__ */ new Map(), "", ga, null, r);
}
function Qt(e, t, r, n, i = null, a = !1) {
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
    if (Yr(e)) {
      var s = (
        /** @type {Snapshot<any>} */
        Array(e.length)
      );
      t.set(e, s), i !== null && t.set(i, s);
      for (var c = 0; c < e.length; c += 1) {
        var f = e[c];
        c in e && (s[c] = Qt(f, t, r, n, null, a));
      }
      return s;
    }
    if (Un(e) === Pi) {
      s = {}, t.set(e, s), i !== null && t.set(i, s);
      for (var h of Object.keys(e))
        s[h] = Qt(
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
      return Qt(
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
function rn(e, t, r) {
  if (e == null)
    return t(void 0), r && r(void 0), et;
  const n = ue(
    () => e.subscribe(
      t,
      // @ts-expect-error
      r
    )
  );
  return n.unsubscribe ? () => n.unsubscribe() : n;
}
const vt = [];
function _a(e, t) {
  return {
    subscribe: Vt(e, t).subscribe
  };
}
function Vt(e, t = et) {
  let r = null;
  const n = /* @__PURE__ */ new Set();
  function i(s) {
    if (Ii(e, s) && (e = s, r)) {
      const c = !vt.length;
      for (const f of n)
        f[1](), vt.push(f, e);
      if (c) {
        for (let f = 0; f < vt.length; f += 2)
          vt[f][0](vt[f + 1]);
        vt.length = 0;
      }
    }
  }
  function a(s) {
    i(s(
      /** @type {T} */
      e
    ));
  }
  function o(s, c = et) {
    const f = [s, c];
    return n.add(f), n.size === 1 && (r = t(i, a) || et), s(
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
  return _a(r, (o, s) => {
    let c = !1;
    const f = [];
    let h = 0, v = et;
    const m = () => {
      if (h)
        return;
      v();
      const _ = t(n ? f[0] : f, o, s);
      a ? o(_) : v = typeof _ == "function" ? _ : et;
    }, y = i.map(
      (_, S) => rn(
        _,
        (B) => {
          f[S] = B, h &= ~(1 << S), c && m();
        },
        () => {
          h |= 1 << S;
        }
      )
    );
    return c = !0, m(), function() {
      Mr(y), v(), c = !1;
    };
  });
}
function ya(e) {
  let t;
  return rn(e, (r) => t = r)(), t;
}
let Zt = !1, Cr = /* @__PURE__ */ Symbol("unmounted");
function cn(e, t, r) {
  const n = r[t] ??= {
    store: null,
    source: st(void 0),
    unsubscribe: et
  };
  if (n.store !== e && !(Cr in r))
    if (n.unsubscribe(), n.store = e ?? null, e == null)
      n.source.v = void 0, n.unsubscribe = et;
    else {
      var i = !0;
      n.unsubscribe = rn(e, (a) => {
        i ? n.source.v = a : w(n.source, a);
      }), i = !1;
    }
  return e && Cr in r ? ya(e) : l(n.source);
}
function xa() {
  const e = {};
  function t() {
    Jr(() => {
      for (var r in e)
        e[r].unsubscribe();
      Bi(e, Cr, {
        enumerable: !1,
        value: !0
      });
    });
  }
  return [e, t];
}
function Ea(e) {
  var t = Zt;
  try {
    return Zt = !1, [e(), Zt];
  } finally {
    Zt = t;
  }
}
function wa(e, t) {
  if (t) {
    const r = document.body;
    e.autofocus = !0, Oi(() => {
      document.activeElement === r && e.focus();
    });
  }
}
const Ta = (
  // We gotta write it like this because after downleveling the pure comment may end up in the wrong location
  globalThis?.window?.trustedTypes && /* @__PURE__ */ globalThis.window.trustedTypes.createPolicy("svelte-trusted-html", {
    /** @param {string} html */
    createHTML: (e) => e
  })
);
function Sa(e) {
  return (
    /** @type {string} */
    Ta?.createHTML(e) ?? e
  );
}
function Jn(e) {
  var t = Fn("template");
  return t.innerHTML = Sa(e.replaceAll("<!>", "<!---->")), t.content;
}
function Et(e, t) {
  var r = (
    /** @type {Effect} */
    ar
  );
  r.nodes === null && (r.nodes = { start: e, end: t, a: null, t: null });
}
// @__NO_SIDE_EFFECTS__
function pe(e, t) {
  var r = (t & Li) !== 0, n = (t & Ni) !== 0, i, a = !e.startsWith("<!>");
  return () => {
    i === void 0 && (i = Jn(a ? e : "<!>" + e), r || (i = /** @type {TemplateNode} */
    er(i)));
    var o = (
      /** @type {TemplateNode} */
      n || Mi ? document.importNode(i, !0) : i.cloneNode(!0)
    );
    if (r) {
      var s = (
        /** @type {TemplateNode} */
        er(o)
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
function Aa(e, t, r = "svg") {
  var n = !e.startsWith("<!>"), i = `<${r}>${n ? e : "<!>" + e}</${r}>`, a;
  return () => {
    if (!a) {
      var o = (
        /** @type {DocumentFragment} */
        Jn(i)
      ), s = (
        /** @type {Element} */
        er(o)
      );
      a = /** @type {Element} */
      er(s);
    }
    var c = (
      /** @type {TemplateNode} */
      a.cloneNode(!0)
    );
    return Et(c, c), c;
  };
}
// @__NO_SIDE_EFFECTS__
function Qn(e, t) {
  return /* @__PURE__ */ Aa(e, t, "svg");
}
function Ue(e = "") {
  {
    var t = lt(e + "");
    return Et(t, t), t;
  }
}
function _t() {
  var e = document.createDocumentFragment(), t = document.createComment(""), r = lt();
  return e.append(t, r), Et(t, r), e;
}
function D(e, t) {
  e !== null && e.before(
    /** @type {Node} */
    t
  );
}
class ur {
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
        tr(n), this.#n.delete(r);
      else {
        var i = this.#e.get(r);
        i && (tr(i.effect), this.#r.set(r, i.effect), this.#e.delete(r), i.fragment.lastChild.remove(), this.anchor.before(i.fragment), n = i.effect);
      }
      for (const [a, o] of this.#t) {
        if (this.#t.delete(a), a === t)
          break;
        const s = this.#e.get(o);
        s && (ot(s.effect), this.#e.delete(o));
      }
      for (const [a, o] of this.#r) {
        if (a === r || this.#n.has(a)) continue;
        const s = () => {
          if (Array.from(this.#t.values()).includes(a)) {
            var f = document.createDocumentFragment();
            jn(o, f), f.append(lt()), this.#e.set(a, { effect: o, fragment: f });
          } else
            ot(o);
          this.#n.delete(a), this.#r.delete(a);
        };
        this.#i || !n ? (this.#n.add(a), Qr(o, s, !1)) : s();
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
      r.includes(n) || (ot(i.effect), this.#e.delete(n));
  };
  /**
   *
   * @param {any} key
   * @param {null | ((target: TemplateNode) => void)} fn
   */
  ensure(t, r) {
    var n = (
      /** @type {Batch} */
      Gn
    ), i = Vn();
    if (r && !this.#r.has(t) && !this.#e.has(t))
      if (i) {
        var a = document.createDocumentFragment(), o = lt();
        a.append(o), this.#e.set(t, {
          effect: ut(() => r(o)),
          fragment: a
        });
      } else
        this.#r.set(
          t,
          ut(() => r(this.anchor))
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
function Ha(e, t, ...r) {
  var n = new ur(e);
  Gt(() => {
    const i = t() ?? null;
    n.ensure(i, i && ((a) => i(a, ...r)));
  }, sr);
}
function Pa(e) {
  yt === null && Yn(), zn && yt.l !== null ? Ba(yt).m.push(e) : Pe(() => {
    const t = ue(e);
    if (typeof t == "function") return (
      /** @type {() => void} */
      t
    );
  });
}
function Ia(e) {
  yt === null && Yn(), Pa(() => () => ue(e));
}
function Ba(e) {
  var t = (
    /** @type {ComponentContextLegacy} */
    e.l
  );
  return t.u ??= { a: [], b: [], m: [] };
}
function ee(e, t, r = !1) {
  var n = new ur(e), i = r ? sr : 0;
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
function hn(e, t) {
  return t;
}
function Oa(e, t, r) {
  for (var n = [], i = t.length, a, o = t.length, s = 0; s < i; s++) {
    let v = t[s];
    Qr(
      v,
      () => {
        if (a) {
          if (a.pending.delete(v), a.done.add(v), a.pending.size === 0) {
            var m = (
              /** @type {Set<EachOutroGroup>} */
              e.outrogroups
            );
            Rr(e, Kr(a.done)), m.delete(a), m.size === 0 && (e.outrogroups = null);
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
      Fi(h), h.append(f), e.items.clear();
    }
    Rr(e, t, !c);
  } else
    a = {
      pending: new Set(t),
      done: /* @__PURE__ */ new Set()
    }, (e.outrogroups ??= /* @__PURE__ */ new Set()).add(a);
}
function Rr(e, t, r = !0) {
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
      a.f |= Xe;
      const o = document.createDocumentFragment();
      jn(a, o);
    } else
      ot(t[i], r);
  }
}
var dn;
function pn(e, t, r, n, i, a = null) {
  var o = e, s = /* @__PURE__ */ new Map(), c = null, f = Xn(() => {
    var p = r();
    return (
      /** @type {V[]} */
      Yr(p) ? p : p == null ? [] : Kr(p)
    );
  }), h, v = /* @__PURE__ */ new Map(), m = !0;
  function y(p) {
    (B.effect.f & qn) === 0 && (B.pending.delete(p), B.fallback = c, Ma(B, h, o, t, n), c !== null && (h.length === 0 ? (c.f & Xe) === 0 ? tr(c) : (c.f ^= Xe, Dt(c, null, o)) : Qr(c, () => {
      c = null;
    })));
  }
  function _(p) {
    B.pending.delete(p);
  }
  var S = Gt(() => {
    h = /** @type {V[]} */
    l(f);
    for (var p = h.length, g = /* @__PURE__ */ new Set(), T = (
      /** @type {Batch} */
      Gn
    ), E = Vn(), x = 0; x < p; x += 1) {
      var H = h[x], P = n(H, x), L = m ? null : s.get(P);
      L ? (L.v && sn(L.v, H), L.i && sn(L.i, x), E && T.unskip_effect(L.e)) : (L = La(
        s,
        m ? o : dn ??= lt(),
        H,
        P,
        x,
        i,
        t,
        r
      ), m || (L.e.f |= Xe), s.set(P, L)), g.add(P);
    }
    if (p === 0 && a && !c && (m ? c = ut(() => a(o)) : (c = ut(() => a(dn ??= lt())), c.f |= Xe)), p > g.size && Ci(), !m)
      if (v.set(T, g), E) {
        for (const [U, G] of s)
          g.has(U) || T.skip_effect(G.e);
        T.oncommit(y), T.ondiscard(_);
      } else
        y(T);
    l(f);
  }), B = { effect: S, items: s, pending: v, outrogroups: null, fallback: c };
  m = !1;
}
function Ct(e) {
  for (; e !== null && (e.f & Ui) === 0; )
    e = e.next;
  return e;
}
function Ma(e, t, r, n, i) {
  var a = t.length, o = e.items, s = Ct(e.effect.first), c, f = null, h = [], v = [], m, y, _, S;
  for (S = 0; S < a; S += 1) {
    if (m = t[S], y = i(m, S), _ = /** @type {EachItem} */
    o.get(y).e, e.outrogroups !== null)
      for (const L of e.outrogroups)
        L.pending.delete(_), L.done.delete(_);
    if ((_.f & _r) !== 0 && tr(_), (_.f & Xe) !== 0)
      if (_.f ^= Xe, _ === s)
        Dt(_, null, r);
      else {
        var B = f ? f.next : s;
        _ === e.effect.last && (e.effect.last = _.prev), _.prev && (_.prev.next = _.next), _.next && (_.next.prev = _.prev), Qe(e, f, _), Qe(e, _, B), Dt(_, B, r), f = _, h = [], v = [], s = Ct(f.next);
        continue;
      }
    if (_ !== s) {
      if (c !== void 0 && c.has(_)) {
        if (h.length < v.length) {
          var p = v[0], g;
          f = p.prev;
          var T = h[0], E = h[h.length - 1];
          for (g = 0; g < h.length; g += 1)
            Dt(h[g], p, r);
          for (g = 0; g < v.length; g += 1)
            c.delete(v[g]);
          Qe(e, T.prev, E.next), Qe(e, f, T), Qe(e, E, p), s = p, f = E, S -= 1, h = [], v = [];
        } else
          c.delete(_), Dt(_, s, r), Qe(e, _.prev, _.next), Qe(e, _, f === null ? e.effect.first : f.next), Qe(e, f, _), f = _;
        continue;
      }
      for (h = [], v = []; s !== null && s !== _; )
        (c ??= /* @__PURE__ */ new Set()).add(s), v.push(s), s = Ct(s.next);
      if (s === null)
        continue;
    }
    (_.f & Xe) === 0 && h.push(_), f = _, s = Ct(_.next);
  }
  if (e.outrogroups !== null) {
    for (const L of e.outrogroups)
      L.pending.size === 0 && (Rr(e, Kr(L.done)), e.outrogroups?.delete(L));
    e.outrogroups.size === 0 && (e.outrogroups = null);
  }
  if (s !== null || c !== void 0) {
    var x = [];
    if (c !== void 0)
      for (_ of c)
        (_.f & _r) === 0 && x.push(_);
    for (; s !== null; )
      (s.f & _r) === 0 && s !== e.fallback && x.push(s), s = Ct(s.next);
    var H = x.length;
    if (H > 0) {
      var P = null;
      Oa(e, x, P);
    }
  }
}
function La(e, t, r, n, i, a, o, s) {
  var c = (o & Di) !== 0 ? (o & ki) === 0 ? st(r, !1, !1) : on(r) : null, f = (o & Ri) !== 0 ? on(i) : null;
  return {
    v: c,
    i: f,
    e: ut(() => (a(t, c ?? r, f ?? i, s), () => {
      e.delete(n);
    }))
  };
}
function Dt(e, t, r) {
  if (e.nodes)
    for (var n = e.nodes.start, i = e.nodes.end, a = t && (t.f & Xe) === 0 ? (
      /** @type {EffectNodes} */
      t.nodes.start
    ) : r; n !== null; ) {
      var o = (
        /** @type {TemplateNode} */
        Gi(n)
      );
      if (a.before(n), n === i)
        return;
      n = o;
    }
}
function Qe(e, t, r) {
  t === null ? e.effect.first = r : t.next = r, r === null ? e.effect.last = t : r.prev = t;
}
function Dr(e, t, r, n, i) {
  var a = t.$$slots?.[r], o = !1;
  a === !0 && (a = t[r === "default" ? "children" : r], o = !0), a === void 0 || a(e, o ? () => n : n);
}
function Na(e, t, r) {
  var n = new ur(e);
  Gt(() => {
    var i = t() ?? null;
    n.ensure(i, i && ((a) => r(a, i)));
  }, sr);
}
const Ca = () => performance.now(), Ne = {
  // don't access requestAnimationFrame eagerly outside method
  // this allows basic testing of user code without JSDOM
  // bunder will eval and remove ternary when the user's app is built
  tick: (
    /** @param {any} _ */
    (e) => requestAnimationFrame(e)
  ),
  now: () => Ca(),
  tasks: /* @__PURE__ */ new Set()
};
function Kn() {
  const e = Ne.now();
  Ne.tasks.forEach((t) => {
    t.c(e) || (Ne.tasks.delete(t), t.f());
  }), Ne.tasks.size !== 0 && Ne.tick(Kn);
}
function Ra(e) {
  let t;
  return Ne.tasks.size === 0 && Ne.tick(Kn), {
    promise: new Promise((r) => {
      Ne.tasks.add(t = { c: e, f: r });
    }),
    abort() {
      Ne.tasks.delete(t);
    }
  };
}
function Da(e, t, r, n, i, a) {
  var o = null, s = (
    /** @type {TemplateNode} */
    e
  ), c = new ur(s, !1);
  Gt(() => {
    const f = t() || null;
    var h = f === "svg" ? ji : void 0;
    if (f === null) {
      c.ensure(null, null);
      return;
    }
    return c.ensure(f, (v) => {
      if (f) {
        if (o = Fn(f, h), Et(o, o), n) {
          var m = null, y = o.appendChild(lt());
          n(o, y), m?.remove();
        }
        ar.nodes.end = o, v.before(o);
      }
    }), () => {
    };
  }, sr), Jr(() => {
  });
}
function ka(e, t) {
  var r = void 0, n;
  Wn(() => {
    r !== (r = t()) && (n && (ot(n), n = null), r && (n = ut(() => {
      $r(() => (
        /** @type {(node: Element) => void} */
        r(e)
      ));
    })));
  });
}
function $n(e) {
  var t, r, n = "";
  if (typeof e == "string" || typeof e == "number") n += e;
  else if (typeof e == "object") if (Array.isArray(e)) {
    var i = e.length;
    for (t = 0; t < i; t++) e[t] && (r = $n(e[t])) && (n && (n += " "), n += r);
  } else for (r in e) e[r] && (n && (n += " "), n += r);
  return n;
}
function Ua() {
  for (var e, t, r = 0, n = "", i = arguments.length; r < i; r++) (e = arguments[r]) && (t = $n(e)) && (n && (n += " "), n += t);
  return n;
}
function Fa(e) {
  return typeof e == "object" ? Ua(e) : e ?? "";
}
const mn = [...` 	
\r\f \v\uFEFF`];
function Ga(e, t, r) {
  var n = e == null ? "" : "" + e;
  if (t && (n = n ? n + " " + t : t), r) {
    for (var i of Object.keys(r))
      if (r[i])
        n = n ? n + " " + i : i;
      else if (n.length)
        for (var a = i.length, o = 0; (o = n.indexOf(i, o)) >= 0; ) {
          var s = o + a;
          (o === 0 || mn.includes(n[o - 1])) && (s === n.length || mn.includes(n[s])) ? n = (o === 0 ? "" : n.substring(0, o)) + n.substring(s + 1) : o = s;
        }
  }
  return n === "" ? null : n;
}
function vn(e, t = !1) {
  var r = t ? " !important;" : ";", n = "";
  for (var i of Object.keys(e)) {
    var a = e[i];
    a != null && a !== "" && (n += " " + i + ": " + a + r);
  }
  return n;
}
function yr(e) {
  return e[0] !== "-" || e[1] !== "-" ? e.toLowerCase() : e;
}
function ja(e, t) {
  if (t) {
    var r = "", n, i;
    if (Array.isArray(t) ? (n = t[0], i = t[1]) : n = t, e) {
      e = String(e).replaceAll(/\/\*.*?\*\//g, "").trim();
      var a = !1, o = 0, s = !1, c = [];
      n && c.push(...Object.keys(n).map(yr)), i && c.push(...Object.keys(i).map(yr));
      var f = 0, h = -1;
      const S = e.length;
      for (var v = 0; v < S; v++) {
        var m = e[v];
        if (s ? m === "/" && e[v - 1] === "*" && (s = !1) : a ? a === m && (a = !1) : m === "/" && e[v + 1] === "*" ? s = !0 : m === '"' || m === "'" ? a = m : m === "(" ? o++ : m === ")" && o--, !s && a === !1 && o === 0) {
          if (m === ":" && h === -1)
            h = v;
          else if (m === ";" || v === S - 1) {
            if (h !== -1) {
              var y = yr(e.substring(f, h).trim());
              if (!c.includes(y)) {
                m !== ";" && v++;
                var _ = e.substring(f, v).trim();
                r += " " + _ + ";";
              }
            }
            f = v + 1, h = -1;
          }
        }
      }
    }
    return n && (r += vn(n)), i && (r += vn(i, !0)), r = r.trim(), r === "" ? null : r;
  }
  return e == null ? null : String(e);
}
function tt(e, t, r, n, i, a) {
  var o = (
    /** @type {any} */
    e[ln]
  );
  if (o !== r || o === void 0) {
    var s = Ga(r, n, a);
    s == null ? e.removeAttribute("class") : t ? e.className = s : e.setAttribute("class", s), e[ln] = r;
  } else if (a && i !== a)
    for (var c in a) {
      var f = !!a[c];
      (i == null || f !== !!i[c]) && e.classList.toggle(c, f);
    }
  return a;
}
function xr(e, t = {}, r, n) {
  for (var i in r) {
    var a = r[i];
    t[i] !== a && (r[i] == null ? e.style.removeProperty(i) : e.style.setProperty(i, a, n));
  }
}
function Ce(e, t, r, n) {
  var i = (
    /** @type {any} */
    e[un]
  );
  if (i !== t) {
    var a = ja(t, n);
    a == null ? e.removeAttribute("style") : e.style.cssText = a, e[un] = t;
  } else n && (Array.isArray(n) ? (xr(e, r?.[0], n[0]), xr(e, r?.[1], n[1], "important")) : xr(e, r, n));
  return n;
}
function kr(e, t, r = !1) {
  if (e.multiple) {
    if (t == null)
      return;
    if (!Yr(t))
      return Vi();
    for (var n of e.options)
      n.selected = t.includes(gn(n));
    return;
  }
  for (n of e.options) {
    var i = gn(n);
    if (zi(i, t)) {
      n.selected = !0;
      return;
    }
  }
  (!r || t !== void 0) && (e.selectedIndex = -1);
}
function Va(e) {
  var t = new MutationObserver(() => {
    "__value" in e && kr(e, e.__value);
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
  }), Jr(() => {
    t.disconnect();
  });
}
function gn(e) {
  return "__value" in e ? e.__value : e.value;
}
const kt = /* @__PURE__ */ Symbol("class"), bt = /* @__PURE__ */ Symbol("style"), ei = /* @__PURE__ */ Symbol("is custom element"), ti = /* @__PURE__ */ Symbol("is html"), za = en ? "input" : "INPUT", Xa = en ? "option" : "OPTION", qa = en ? "select" : "SELECT";
function Wa(e, t) {
  t ? e.hasAttribute("selected") || e.setAttribute("selected", "") : e.removeAttribute("selected");
}
function xt(e, t, r, n) {
  var i = ri(e);
  i[t] !== (i[t] = r) && (t === "loading" && (e[Xi] = r), r == null ? e.removeAttribute(t) : typeof r != "string" && ni(e).includes(t) ? e[t] = r : e.setAttribute(t, r));
}
function Za(e, t, r, n, i = !1, a = !1) {
  var o = ri(e), s = o[ei], c = !o[ti], f = t || {}, h = e.nodeName === Xa;
  for (var v in t)
    !(v in r) && v[0] + v[1] !== "$$" && (r[v] = null);
  r.class ? r.class = Fa(r.class) : r.class = null, r[bt] && (r.style ??= null);
  var m = ni(e);
  if (e.nodeName === za && "type" in r && ("value" in r || "__value" in r)) {
    var y = r.type;
    (y !== f.type || y === void 0 && e.hasAttribute("type")) && (f.type = y, xt(e, "type", y));
  }
  for (const E in r) {
    let x = r[E];
    if (h && E === "value" && x == null) {
      e.value = e.__value = "", f[E] = x;
      continue;
    }
    if (E === "class") {
      var _ = e.namespaceURI === "http://www.w3.org/1999/xhtml";
      tt(e, _, x, n, t?.[kt], r[kt]), f[E] = x, f[kt] = r[kt];
      continue;
    }
    if (E === "style") {
      Ce(e, x, t?.[bt], r[bt]), f[E] = x, f[bt] = r[bt];
      continue;
    }
    var S = f[E];
    if (!(x === S && !(x === void 0 && e.hasAttribute(E)))) {
      f[E] = x;
      var B = E[0] + E[1];
      if (B !== "$$")
        if (B === "on") {
          const H = {}, P = "$$" + E;
          let L = E.slice(2);
          var p = ta(L);
          if (Qi(L) && (L = L.slice(0, -7), H.capture = !0), !p && S) {
            if (x != null) continue;
            e.removeEventListener(L, f[P], H), f[P] = null;
          }
          if (p)
            Ke(L, e, x), jt([L]);
          else if (x != null) {
            let U = function(G) {
              f[E].call(this, G);
            };
            f[P] = Ki(L, e, U, H);
          }
        } else if (E === "style")
          xt(e, E, x);
        else if (E === "autofocus")
          wa(
            /** @type {HTMLElement} */
            e,
            !!x
          );
        else if (!s && (E === "__value" || E === "value" && x != null))
          e.value = e.__value = x;
        else if (E === "selected" && h)
          Wa(
            /** @type {HTMLOptionElement} */
            e,
            x
          );
        else {
          var g = E;
          c || (g = $i(g));
          var T = g === "defaultValue" || g === "defaultChecked";
          if (x == null && !s && !T)
            if (o[E] = null, g === "value" || g === "checked") {
              let H = (
                /** @type {HTMLInputElement} */
                e
              );
              const P = t === void 0;
              if (g === "value") {
                let L = H.defaultValue;
                H.removeAttribute(g), H.defaultValue = L, H.value = H.__value = P ? L : null;
              } else {
                let L = H.defaultChecked;
                H.removeAttribute(g), H.defaultChecked = L, H.checked = P ? L : !1;
              }
            } else
              e.removeAttribute(E);
          else T || m.includes(g) && (s || typeof x != "string") ? (e[g] = x, g in o && (o[g] = ea)) : typeof x != "function" && xt(e, g, x);
        }
    }
  }
  return f;
}
function Ya(e, t, r = [], n = [], i = [], a, o = !1, s = !1) {
  Yi(i, r, n, (c) => {
    var f = void 0, h = {}, v = e.nodeName === qa, m = !1;
    if (Wn(() => {
      var _ = t(...c.map(l)), S = Za(
        e,
        f,
        _,
        a,
        o,
        s
      );
      m && v && "value" in _ && kr(
        /** @type {HTMLSelectElement} */
        e,
        _.value
      );
      for (let p of Object.getOwnPropertySymbols(h))
        _[p] || ot(h[p]);
      for (let p of Object.getOwnPropertySymbols(_)) {
        var B = _[p];
        p.description === Ji && (!f || B !== f[p]) && (h[p] && ot(h[p]), h[p] = ut(() => ka(e, () => B))), S[p] = B;
      }
      f = S;
    }), v) {
      var y = (
        /** @type {HTMLSelectElement} */
        e
      );
      $r(() => {
        kr(
          y,
          /** @type {Record<string | symbol, any>} */
          f.value,
          !0
        ), Va(y);
      });
    }
    m = !0;
  });
}
function ri(e) {
  return (
    /** @type {Record<string | symbol, unknown>} **/
    /** @type {any} */
    e[qi] ??= {
      [ei]: e.nodeName.includes("-"),
      [ti]: e.namespaceURI === Wi
    }
  );
}
var bn = /* @__PURE__ */ new Map();
function ni(e) {
  var t = e.getAttribute("is") || e.nodeName, r = bn.get(t);
  if (r) return r;
  bn.set(t, r = []);
  for (var n, i = e, a = Element.prototype; a !== i; ) {
    n = Zi(i);
    for (var o in n)
      n[o].set && // better safe than sorry, we don't want spread attributes to mess with HTML content
      o !== "innerHTML" && o !== "textContent" && o !== "innerText" && r.push(o);
    i = Un(i);
  }
  return r;
}
function Er(e, t) {
  return e === t || e?.[tn] === t;
}
function rr(e = {}, t, r, n) {
  var i = (
    /** @type {ComponentContext} */
    yt.r
  ), a = (
    /** @type {Effect} */
    ar
  );
  return $r(() => {
    var o, s;
    return ra(() => {
      o = s, s = [], ue(() => {
        Er(r(...s), e) || (t(e, ...s), o && Er(r(...o), e) && t(null, ...o));
      });
    }), () => {
      let c = a;
      for (; c !== i && c.parent !== null && c.parent.f & na; )
        c = c.parent;
      const f = () => {
        s && Er(r(...s), e) && t(null, ...s);
      }, h = c.teardown;
      c.teardown = () => {
        f(), h?.();
      };
    };
  }), e;
}
function Ja(e = !1) {
  const t = (
    /** @type {ComponentContextLegacy} */
    yt
  ), r = t.l.u;
  if (!r) return;
  let n = () => He(t.s);
  if (e) {
    let i = 0, a = (
      /** @type {Record<string, any>} */
      {}
    );
    const o = Lr(() => {
      let s = !1;
      const c = t.s;
      for (const f in c)
        c[f] !== a[f] && (a[f] = c[f], s = !0);
      return s && i++, i;
    });
    n = () => l(o);
  }
  r.b.length && ia(() => {
    _n(t, n), Mr(r.b);
  }), Pe(() => {
    const i = ue(() => r.m.map(aa));
    return () => {
      for (const a of i)
        typeof a == "function" && a();
    };
  }), r.a.length && Pe(() => {
    _n(t, n), Mr(r.a);
  });
}
function _n(e, t) {
  if (e.l.s)
    for (const r of e.l.s) l(r);
  t();
}
const Qa = {
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
function Ka(e, t, r) {
  return new Proxy({ props: e, exclude: t }, Qa);
}
const $a = {
  get(e, t) {
    let r = e.props.length;
    for (; r--; ) {
      let n = e.props[r];
      if (Nt(n) && (n = n()), typeof n == "object" && n !== null && t in n) return n[t];
    }
  },
  set(e, t, r) {
    let n = e.props.length;
    for (; n--; ) {
      let i = e.props[n];
      Nt(i) && (i = i());
      const a = Nr(i, t);
      if (a && a.set)
        return a.set(r), !0;
    }
    return !1;
  },
  getOwnPropertyDescriptor(e, t) {
    let r = e.props.length;
    for (; r--; ) {
      let n = e.props[r];
      if (Nt(n) && (n = n()), typeof n == "object" && n !== null && t in n) {
        const i = Nr(n, t);
        return i && !i.configurable && (i.configurable = !0), i;
      }
    }
  },
  has(e, t) {
    if (t === tn || t === Zn) return !1;
    for (let r of e.props)
      if (Nt(r) && (r = r()), r != null && t in r) return !0;
    return !1;
  },
  ownKeys(e) {
    const t = [];
    for (let r of e.props)
      if (Nt(r) && (r = r()), !!r) {
        for (const n in r)
          t.includes(n) || t.push(n);
        for (const n of Object.getOwnPropertySymbols(r))
          t.includes(n) || t.push(n);
      }
    return t;
  }
};
function es(...e) {
  return new Proxy({ props: e }, $a);
}
function O(e, t, r, n) {
  var i = !zn || (r & ua) !== 0, a = (r & la) !== 0, o = (r & ca) !== 0, s = (
    /** @type {V} */
    n
  ), c = !0, f = (
    /** @type {Derived<V> | undefined} */
    void 0
  ), h = () => o && i ? (f ??= Lr(
    /** @type {() => V} */
    n
  ), l(f)) : (c && (c = !1, s = o ? ue(
    /** @type {() => V} */
    n
  ) : (
    /** @type {V} */
    n
  )), s);
  let v;
  if (a) {
    var m = tn in e || Zn in e;
    v = Nr(e, t)?.set ?? (m && t in e ? (E) => e[t] = E : void 0);
  }
  var y, _ = !1;
  a ? [y, _] = Ea(() => (
    /** @type {V} */
    e[t]
  )) : y = /** @type {V} */
  e[t], y === void 0 && n !== void 0 && (y = h(), v && (i && sa(), v(y)));
  var S;
  if (i ? S = () => {
    var E = (
      /** @type {V} */
      e[t]
    );
    return E === void 0 ? h() : (c = !0, E);
  } : S = () => {
    var E = (
      /** @type {V} */
      e[t]
    );
    return E !== void 0 && (s = /** @type {V} */
    void 0), E === void 0 ? s : E;
  }, i && (r & oa) === 0)
    return S;
  if (v) {
    var B = e.$$legacy;
    return (
      /** @type {() => V} */
      (function(E, x) {
        return arguments.length > 0 ? ((!i || !x || B || _) && v(x ? S() : E), E) : S();
      })
    );
  }
  var p = !1, g = ((r & fa) !== 0 ? Lr : Xn)(() => (p = !1, S()));
  a && l(g);
  var T = (
    /** @type {Effect} */
    ar
  );
  return (
    /** @type {() => V} */
    (function(E, x) {
      if (arguments.length > 0) {
        const H = x ? l(g) : i && a ? Ut(E) : E;
        return w(g, H), p = !0, s !== void 0 && (s = H), E;
      }
      return ha && p || (T.f & qn) !== 0 ? g.v : l(g);
    })
  );
}
const ts = [
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
], yn = {
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
ts.reduce((e, { color: t, primary: r, secondary: n }) => ({
  ...e,
  [t]: {
    primary: yn[t][r],
    secondary: yn[t][n]
  }
}), {});
function rs(e) {
  return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, "default") ? e.default : e;
}
var wr, xn;
function ns() {
  if (xn) return wr;
  xn = 1;
  var e = function(g) {
    return t(g) && !r(g);
  };
  function t(p) {
    return !!p && typeof p == "object";
  }
  function r(p) {
    var g = Object.prototype.toString.call(p);
    return g === "[object RegExp]" || g === "[object Date]" || a(p);
  }
  var n = typeof Symbol == "function" && Symbol.for, i = n ? /* @__PURE__ */ Symbol.for("react.element") : 60103;
  function a(p) {
    return p.$$typeof === i;
  }
  function o(p) {
    return Array.isArray(p) ? [] : {};
  }
  function s(p, g) {
    return g.clone !== !1 && g.isMergeableObject(p) ? S(o(p), p, g) : p;
  }
  function c(p, g, T) {
    return p.concat(g).map(function(E) {
      return s(E, T);
    });
  }
  function f(p, g) {
    if (!g.customMerge)
      return S;
    var T = g.customMerge(p);
    return typeof T == "function" ? T : S;
  }
  function h(p) {
    return Object.getOwnPropertySymbols ? Object.getOwnPropertySymbols(p).filter(function(g) {
      return Object.propertyIsEnumerable.call(p, g);
    }) : [];
  }
  function v(p) {
    return Object.keys(p).concat(h(p));
  }
  function m(p, g) {
    try {
      return g in p;
    } catch {
      return !1;
    }
  }
  function y(p, g) {
    return m(p, g) && !(Object.hasOwnProperty.call(p, g) && Object.propertyIsEnumerable.call(p, g));
  }
  function _(p, g, T) {
    var E = {};
    return T.isMergeableObject(p) && v(p).forEach(function(x) {
      E[x] = s(p[x], T);
    }), v(g).forEach(function(x) {
      y(p, x) || (m(p, x) && T.isMergeableObject(g[x]) ? E[x] = f(x, T)(p[x], g[x], T) : E[x] = s(g[x], T));
    }), E;
  }
  function S(p, g, T) {
    T = T || {}, T.arrayMerge = T.arrayMerge || c, T.isMergeableObject = T.isMergeableObject || e, T.cloneUnlessOtherwiseSpecified = s;
    var E = Array.isArray(g), x = Array.isArray(p), H = E === x;
    return H ? E ? T.arrayMerge(p, g, T) : _(p, g, T) : s(g, T);
  }
  S.all = function(g, T) {
    if (!Array.isArray(g))
      throw new Error("first argument should be an array");
    return g.reduce(function(E, x) {
      return S(E, x, T);
    }, {});
  };
  var B = S;
  return wr = B, wr;
}
var is = ns();
const as = /* @__PURE__ */ rs(is);
var Ur = function(e, t) {
  return Ur = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(r, n) {
    r.__proto__ = n;
  } || function(r, n) {
    for (var i in n) Object.prototype.hasOwnProperty.call(n, i) && (r[i] = n[i]);
  }, Ur(e, t);
};
function fr(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Class extends value " + String(t) + " is not a constructor or null");
  Ur(e, t);
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
function ss(e, t) {
  var r = {};
  for (var n in e) Object.prototype.hasOwnProperty.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && typeof Object.getOwnPropertySymbols == "function")
    for (var i = 0, n = Object.getOwnPropertySymbols(e); i < n.length; i++)
      t.indexOf(n[i]) < 0 && Object.prototype.propertyIsEnumerable.call(e, n[i]) && (r[n[i]] = e[n[i]]);
  return r;
}
function Tr(e, t, r) {
  if (r || arguments.length === 2) for (var n = 0, i = t.length, a; n < i; n++)
    (a || !(n in t)) && (a || (a = Array.prototype.slice.call(t, 0, n)), a[n] = t[n]);
  return e.concat(a || Array.prototype.slice.call(t));
}
function Sr(e, t) {
  var r = t && t.cache ? t.cache : ds, n = t && t.serializer ? t.serializer : cs, i = t && t.strategy ? t.strategy : us;
  return i(e, {
    cache: r,
    serializer: n
  });
}
function os(e) {
  return e == null || typeof e == "number" || typeof e == "boolean";
}
function ls(e, t, r, n) {
  var i = os(n) ? n : r(n), a = t.get(i);
  return typeof a > "u" && (a = e.call(this, n), t.set(i, a)), a;
}
function ii(e, t, r) {
  var n = Array.prototype.slice.call(arguments, 3), i = r(n), a = t.get(i);
  return typeof a > "u" && (a = e.apply(this, n), t.set(i, a)), a;
}
function ai(e, t, r, n, i) {
  return r.bind(t, e, n, i);
}
function us(e, t) {
  var r = e.length === 1 ? ls : ii;
  return ai(e, this, r, t.cache.create(), t.serializer);
}
function fs(e, t) {
  return ai(e, this, ii, t.cache.create(), t.serializer);
}
var cs = function() {
  return JSON.stringify(arguments);
}, hs = (
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
), ds = {
  create: function() {
    return new hs();
  }
}, Ar = {
  variadic: fs
}, R;
(function(e) {
  e[e.EXPECT_ARGUMENT_CLOSING_BRACE = 1] = "EXPECT_ARGUMENT_CLOSING_BRACE", e[e.EMPTY_ARGUMENT = 2] = "EMPTY_ARGUMENT", e[e.MALFORMED_ARGUMENT = 3] = "MALFORMED_ARGUMENT", e[e.EXPECT_ARGUMENT_TYPE = 4] = "EXPECT_ARGUMENT_TYPE", e[e.INVALID_ARGUMENT_TYPE = 5] = "INVALID_ARGUMENT_TYPE", e[e.EXPECT_ARGUMENT_STYLE = 6] = "EXPECT_ARGUMENT_STYLE", e[e.INVALID_NUMBER_SKELETON = 7] = "INVALID_NUMBER_SKELETON", e[e.INVALID_DATE_TIME_SKELETON = 8] = "INVALID_DATE_TIME_SKELETON", e[e.EXPECT_NUMBER_SKELETON = 9] = "EXPECT_NUMBER_SKELETON", e[e.EXPECT_DATE_TIME_SKELETON = 10] = "EXPECT_DATE_TIME_SKELETON", e[e.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE = 11] = "UNCLOSED_QUOTE_IN_ARGUMENT_STYLE", e[e.EXPECT_SELECT_ARGUMENT_OPTIONS = 12] = "EXPECT_SELECT_ARGUMENT_OPTIONS", e[e.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE = 13] = "EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE = 14] = "INVALID_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR = 15] = "EXPECT_SELECT_ARGUMENT_SELECTOR", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR = 16] = "EXPECT_PLURAL_ARGUMENT_SELECTOR", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT = 17] = "EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT = 18] = "EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT", e[e.INVALID_PLURAL_ARGUMENT_SELECTOR = 19] = "INVALID_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_PLURAL_ARGUMENT_SELECTOR = 20] = "DUPLICATE_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_SELECT_ARGUMENT_SELECTOR = 21] = "DUPLICATE_SELECT_ARGUMENT_SELECTOR", e[e.MISSING_OTHER_CLAUSE = 22] = "MISSING_OTHER_CLAUSE", e[e.INVALID_TAG = 23] = "INVALID_TAG", e[e.INVALID_TAG_NAME = 25] = "INVALID_TAG_NAME", e[e.UNMATCHED_CLOSING_TAG = 26] = "UNMATCHED_CLOSING_TAG", e[e.UNCLOSED_TAG = 27] = "UNCLOSED_TAG";
})(R || (R = {}));
var Q;
(function(e) {
  e[e.literal = 0] = "literal", e[e.argument = 1] = "argument", e[e.number = 2] = "number", e[e.date = 3] = "date", e[e.time = 4] = "time", e[e.select = 5] = "select", e[e.plural = 6] = "plural", e[e.pound = 7] = "pound", e[e.tag = 8] = "tag";
})(Q || (Q = {}));
var wt;
(function(e) {
  e[e.number = 0] = "number", e[e.dateTime = 1] = "dateTime";
})(wt || (wt = {}));
function En(e) {
  return e.type === Q.literal;
}
function ps(e) {
  return e.type === Q.argument;
}
function si(e) {
  return e.type === Q.number;
}
function oi(e) {
  return e.type === Q.date;
}
function li(e) {
  return e.type === Q.time;
}
function ui(e) {
  return e.type === Q.select;
}
function fi(e) {
  return e.type === Q.plural;
}
function ms(e) {
  return e.type === Q.pound;
}
function ci(e) {
  return e.type === Q.tag;
}
function hi(e) {
  return !!(e && typeof e == "object" && e.type === wt.number);
}
function Fr(e) {
  return !!(e && typeof e == "object" && e.type === wt.dateTime);
}
var di = /[ \xA0\u1680\u2000-\u200A\u202F\u205F\u3000]/, vs = /(?:[Eec]{1,6}|G{1,5}|[Qq]{1,5}|(?:[yYur]+|U{1,5})|[ML]{1,5}|d{1,2}|D{1,3}|F{1}|[abB]{1,5}|[hkHK]{1,2}|w{1,2}|W{1}|m{1,2}|s{1,2}|[zZOvVxX]{1,4})(?=([^']*'[^']*')*[^']*$)/g;
function gs(e) {
  var t = {};
  return e.replace(vs, function(r) {
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
var bs = /[\t-\r \x85\u200E\u200F\u2028\u2029]/i;
function _s(e) {
  if (e.length === 0)
    throw new Error("Number skeleton cannot be empty");
  for (var t = e.split(bs).filter(function(m) {
    return m.length > 0;
  }), r = [], n = 0, i = t; n < i.length; n++) {
    var a = i[n], o = a.split("/");
    if (o.length === 0)
      throw new Error("Invalid number skeleton");
    for (var s = o[0], c = o.slice(1), f = 0, h = c; f < h.length; f++) {
      var v = h[f];
      if (v.length === 0)
        throw new Error("Invalid number skeleton");
    }
    r.push({ stem: s, options: c });
  }
  return r;
}
function ys(e) {
  return e.replace(/^(.*?)-/, "");
}
var wn = /^\.(?:(0+)(\*)?|(#+)|(0+)(#+))$/g, pi = /^(@+)?(\+|#+)?[rs]?$/g, xs = /(\*)(0+)|(#+)(0+)|(0+)/g, mi = /^(0+)$/;
function Tn(e) {
  var t = {};
  return e[e.length - 1] === "r" ? t.roundingPriority = "morePrecision" : e[e.length - 1] === "s" && (t.roundingPriority = "lessPrecision"), e.replace(pi, function(r, n, i) {
    return typeof i != "string" ? (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length) : i === "+" ? t.minimumSignificantDigits = n.length : n[0] === "#" ? t.maximumSignificantDigits = n.length : (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length + (typeof i == "string" ? i.length : 0)), "";
  }), t;
}
function vi(e) {
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
function Es(e) {
  var t;
  if (e[0] === "E" && e[1] === "E" ? (t = {
    notation: "engineering"
  }, e = e.slice(2)) : e[0] === "E" && (t = {
    notation: "scientific"
  }, e = e.slice(1)), t) {
    var r = e.slice(0, 2);
    if (r === "+!" ? (t.signDisplay = "always", e = e.slice(2)) : r === "+?" && (t.signDisplay = "exceptZero", e = e.slice(2)), !mi.test(e))
      throw new Error("Malformed concise eng/scientific notation");
    t.minimumIntegerDigits = e.length;
  }
  return t;
}
function Sn(e) {
  var t = {}, r = vi(e);
  return r || t;
}
function ws(e) {
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
        t.style = "unit", t.unit = ys(i.options[0]);
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
          return F(F({}, c), Sn(f));
        }, {}));
        continue;
      case "engineering":
        t = F(F(F({}, t), { notation: "engineering" }), i.options.reduce(function(c, f) {
          return F(F({}, c), Sn(f));
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
        i.options[0].replace(xs, function(c, f, h, v, m, y) {
          if (f)
            t.minimumIntegerDigits = h.length;
          else {
            if (v && m)
              throw new Error("We currently do not support maximum integer digits");
            if (y)
              throw new Error("We currently do not support exact integer digits");
          }
          return "";
        });
        continue;
    }
    if (mi.test(i.stem)) {
      t.minimumIntegerDigits = i.stem.length;
      continue;
    }
    if (wn.test(i.stem)) {
      if (i.options.length > 1)
        throw new RangeError("Fraction-precision stems only accept a single optional option");
      i.stem.replace(wn, function(c, f, h, v, m, y) {
        return h === "*" ? t.minimumFractionDigits = f.length : v && v[0] === "#" ? t.maximumFractionDigits = v.length : m && y ? (t.minimumFractionDigits = m.length, t.maximumFractionDigits = m.length + y.length) : (t.minimumFractionDigits = f.length, t.maximumFractionDigits = f.length), "";
      });
      var a = i.options[0];
      a === "w" ? t = F(F({}, t), { trailingZeroDisplay: "stripIfInteger" }) : a && (t = F(F({}, t), Tn(a)));
      continue;
    }
    if (pi.test(i.stem)) {
      t = F(F({}, t), Tn(i.stem));
      continue;
    }
    var o = vi(i.stem);
    o && (t = F(F({}, t), o));
    var s = Es(i.stem);
    s && (t = F(F({}, t), s));
  }
  return t;
}
var Yt = {
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
function Ts(e, t) {
  for (var r = "", n = 0; n < e.length; n++) {
    var i = e.charAt(n);
    if (i === "j") {
      for (var a = 0; n + 1 < e.length && e.charAt(n + 1) === i; )
        a++, n++;
      var o = 1 + (a & 1), s = a < 2 ? 1 : 3 + (a >> 1), c = "a", f = Ss(t);
      for ((f == "H" || f == "k") && (s = 0); s-- > 0; )
        r += c;
      for (; o-- > 0; )
        r = f + r;
    } else i === "J" ? r += "H" : r += i;
  }
  return r;
}
function Ss(e) {
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
  var i = Yt[n || ""] || Yt[r || ""] || Yt["".concat(r, "-001")] || Yt["001"];
  return i[0];
}
var Hr, As = new RegExp("^".concat(di.source, "*")), Hs = new RegExp("".concat(di.source, "*$"));
function k(e, t) {
  return { start: e, end: t };
}
var Ps = !!String.prototype.startsWith && "_a".startsWith("a", 1), Is = !!String.fromCodePoint, Bs = !!Object.fromEntries, Os = !!String.prototype.codePointAt, Ms = !!String.prototype.trimStart, Ls = !!String.prototype.trimEnd, Ns = !!Number.isSafeInteger, Cs = Ns ? Number.isSafeInteger : function(e) {
  return typeof e == "number" && isFinite(e) && Math.floor(e) === e && Math.abs(e) <= 9007199254740991;
}, Gr = !0;
try {
  var Rs = bi("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  Gr = ((Hr = Rs.exec("a")) === null || Hr === void 0 ? void 0 : Hr[0]) === "a";
} catch {
  Gr = !1;
}
var An = Ps ? (
  // Native
  function(t, r, n) {
    return t.startsWith(r, n);
  }
) : (
  // For IE11
  function(t, r, n) {
    return t.slice(n, n + r.length) === r;
  }
), jr = Is ? String.fromCodePoint : (
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
), Hn = (
  // native
  Bs ? Object.fromEntries : (
    // Ponyfill
    function(t) {
      for (var r = {}, n = 0, i = t; n < i.length; n++) {
        var a = i[n], o = a[0], s = a[1];
        r[o] = s;
      }
      return r;
    }
  )
), gi = Os ? (
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
), Ds = Ms ? (
  // Native
  function(t) {
    return t.trimStart();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(As, "");
  }
), ks = Ls ? (
  // Native
  function(t) {
    return t.trimEnd();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(Hs, "");
  }
);
function bi(e, t) {
  return new RegExp(e, t);
}
var Vr;
if (Gr) {
  var Pn = bi("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  Vr = function(t, r) {
    var n;
    Pn.lastIndex = r;
    var i = Pn.exec(t);
    return (n = i[1]) !== null && n !== void 0 ? n : "";
  };
} else
  Vr = function(t, r) {
    for (var n = []; ; ) {
      var i = gi(t, r);
      if (i === void 0 || _i(i) || js(i))
        break;
      n.push(i), r += i >= 65536 ? 2 : 1;
    }
    return jr.apply(void 0, n);
  };
var Us = (
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
              type: Q.pound,
              location: k(s, this.clonePosition())
            });
          } else if (a === 60 && !this.ignoreTag && this.peek() === 47) {
            if (n)
              break;
            return this.error(R.UNMATCHED_CLOSING_TAG, k(this.clonePosition(), this.clonePosition()));
          } else if (a === 60 && !this.ignoreTag && zr(this.peek() || 0)) {
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
            type: Q.literal,
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
          if (this.isEOF() || !zr(this.char()))
            return this.error(R.INVALID_TAG, k(s, this.clonePosition()));
          var c = this.clonePosition(), f = this.parseTagName();
          return i !== f ? this.error(R.UNMATCHED_CLOSING_TAG, k(c, this.clonePosition())) : (this.bumpSpace(), this.bumpIf(">") ? {
            val: {
              type: Q.tag,
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
      for (this.bump(); !this.isEOF() && Gs(this.char()); )
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
        val: { type: Q.literal, value: i, location: c },
        err: null
      };
    }, e.prototype.tryParseLeftAngleBracket = function() {
      return !this.isEOF() && this.char() === 60 && (this.ignoreTag || // If at the opening tag or closing tag position, bail.
      !Fs(this.peek() || 0)) ? (this.bump(), "<") : null;
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
      return jr.apply(void 0, r);
    }, e.prototype.tryParseUnquoted = function(t, r) {
      if (this.isEOF())
        return null;
      var n = this.char();
      return n === 60 || n === 123 || n === 35 && (r === "plural" || r === "selectordinal") || n === 125 && t > 0 ? null : (this.bump(), jr(n));
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
              type: Q.argument,
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
      var t = this.clonePosition(), r = this.offset(), n = Vr(this.message, r), i = r + n.length;
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
            var h = this.clonePosition(), v = this.parseSimpleArgStyleIfPossible();
            if (v.err)
              return v;
            var m = ks(v.val);
            if (m.length === 0)
              return this.error(R.EXPECT_ARGUMENT_STYLE, k(this.clonePosition(), this.clonePosition()));
            var y = k(h, this.clonePosition());
            f = { style: m, styleLocation: y };
          }
          var _ = this.tryParseArgumentClose(i);
          if (_.err)
            return _;
          var S = k(i, this.clonePosition());
          if (f && An(f?.style, "::", 0)) {
            var B = Ds(f.style.slice(2));
            if (s === "number") {
              var v = this.parseNumberSkeletonFromString(B, f.styleLocation);
              return v.err ? v : {
                val: { type: Q.number, value: n, location: S, style: v.val },
                err: null
              };
            } else {
              if (B.length === 0)
                return this.error(R.EXPECT_DATE_TIME_SKELETON, S);
              var p = B;
              this.locale && (p = Ts(B, this.locale));
              var m = {
                type: wt.dateTime,
                pattern: p,
                location: f.styleLocation,
                parsedOptions: this.shouldParseSkeletons ? gs(p) : {}
              }, g = s === "date" ? Q.date : Q.time;
              return {
                val: { type: g, value: n, location: S, style: m },
                err: null
              };
            }
          }
          return {
            val: {
              type: s === "number" ? Q.number : s === "date" ? Q.date : Q.time,
              value: n,
              location: S,
              style: (a = f?.style) !== null && a !== void 0 ? a : null
            },
            err: null
          };
        }
        case "plural":
        case "selectordinal":
        case "select": {
          var T = this.clonePosition();
          if (this.bumpSpace(), !this.bumpIf(","))
            return this.error(R.EXPECT_SELECT_ARGUMENT_OPTIONS, k(T, F({}, T)));
          this.bumpSpace();
          var E = this.parseIdentifierIfPossible(), x = 0;
          if (s !== "select" && E.value === "offset") {
            if (!this.bumpIf(":"))
              return this.error(R.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, k(this.clonePosition(), this.clonePosition()));
            this.bumpSpace();
            var v = this.tryParseDecimalInteger(R.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, R.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE);
            if (v.err)
              return v;
            this.bumpSpace(), E = this.parseIdentifierIfPossible(), x = v.val;
          }
          var H = this.tryParsePluralOrSelectOptions(t, s, r, E);
          if (H.err)
            return H;
          var _ = this.tryParseArgumentClose(i);
          if (_.err)
            return _;
          var P = k(i, this.clonePosition());
          return s === "select" ? {
            val: {
              type: Q.select,
              value: n,
              options: Hn(H.val),
              location: P
            },
            err: null
          } : {
            val: {
              type: Q.plural,
              value: n,
              options: Hn(H.val),
              offset: x,
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
        n = _s(t);
      } catch {
        return this.error(R.INVALID_NUMBER_SKELETON, r);
      }
      return {
        val: {
          type: wt.number,
          tokens: n,
          location: r,
          parsedOptions: this.shouldParseSkeletons ? ws(n) : {}
        },
        err: null
      };
    }, e.prototype.tryParsePluralOrSelectOptions = function(t, r, n, i) {
      for (var a, o = !1, s = [], c = /* @__PURE__ */ new Set(), f = i.value, h = i.location; ; ) {
        if (f.length === 0) {
          var v = this.clonePosition();
          if (r !== "select" && this.bumpIf("=")) {
            var m = this.tryParseDecimalInteger(R.EXPECT_PLURAL_ARGUMENT_SELECTOR, R.INVALID_PLURAL_ARGUMENT_SELECTOR);
            if (m.err)
              return m;
            h = k(v, this.clonePosition()), f = this.message.slice(v.offset, this.offset());
          } else
            break;
        }
        if (c.has(f))
          return this.error(r === "select" ? R.DUPLICATE_SELECT_ARGUMENT_SELECTOR : R.DUPLICATE_PLURAL_ARGUMENT_SELECTOR, h);
        f === "other" && (o = !0), this.bumpSpace();
        var y = this.clonePosition();
        if (!this.bumpIf("{"))
          return this.error(r === "select" ? R.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT : R.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT, k(this.clonePosition(), this.clonePosition()));
        var _ = this.parseMessage(t + 1, r, n);
        if (_.err)
          return _;
        var S = this.tryParseArgumentClose(y);
        if (S.err)
          return S;
        s.push([
          f,
          {
            value: _.val,
            location: k(y, this.clonePosition())
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
      return a ? (o *= n, Cs(o) ? { val: o, err: null } : this.error(r, c)) : this.error(t, c);
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
      var r = gi(this.message, t);
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
      if (An(this.message, t, this.offset())) {
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
      for (; !this.isEOF() && _i(this.char()); )
        this.bump();
    }, e.prototype.peek = function() {
      if (this.isEOF())
        return null;
      var t = this.char(), r = this.offset(), n = this.message.charCodeAt(r + (t >= 65536 ? 2 : 1));
      return n ?? null;
    }, e;
  })()
);
function zr(e) {
  return e >= 97 && e <= 122 || e >= 65 && e <= 90;
}
function Fs(e) {
  return zr(e) || e === 47;
}
function Gs(e) {
  return e === 45 || e === 46 || e >= 48 && e <= 57 || e === 95 || e >= 97 && e <= 122 || e >= 65 && e <= 90 || e == 183 || e >= 192 && e <= 214 || e >= 216 && e <= 246 || e >= 248 && e <= 893 || e >= 895 && e <= 8191 || e >= 8204 && e <= 8205 || e >= 8255 && e <= 8256 || e >= 8304 && e <= 8591 || e >= 11264 && e <= 12271 || e >= 12289 && e <= 55295 || e >= 63744 && e <= 64975 || e >= 65008 && e <= 65533 || e >= 65536 && e <= 983039;
}
function _i(e) {
  return e >= 9 && e <= 13 || e === 32 || e === 133 || e >= 8206 && e <= 8207 || e === 8232 || e === 8233;
}
function js(e) {
  return e >= 33 && e <= 35 || e === 36 || e >= 37 && e <= 39 || e === 40 || e === 41 || e === 42 || e === 43 || e === 44 || e === 45 || e >= 46 && e <= 47 || e >= 58 && e <= 59 || e >= 60 && e <= 62 || e >= 63 && e <= 64 || e === 91 || e === 92 || e === 93 || e === 94 || e === 96 || e === 123 || e === 124 || e === 125 || e === 126 || e === 161 || e >= 162 && e <= 165 || e === 166 || e === 167 || e === 169 || e === 171 || e === 172 || e === 174 || e === 176 || e === 177 || e === 182 || e === 187 || e === 191 || e === 215 || e === 247 || e >= 8208 && e <= 8213 || e >= 8214 && e <= 8215 || e === 8216 || e === 8217 || e === 8218 || e >= 8219 && e <= 8220 || e === 8221 || e === 8222 || e === 8223 || e >= 8224 && e <= 8231 || e >= 8240 && e <= 8248 || e === 8249 || e === 8250 || e >= 8251 && e <= 8254 || e >= 8257 && e <= 8259 || e === 8260 || e === 8261 || e === 8262 || e >= 8263 && e <= 8273 || e === 8274 || e === 8275 || e >= 8277 && e <= 8286 || e >= 8592 && e <= 8596 || e >= 8597 && e <= 8601 || e >= 8602 && e <= 8603 || e >= 8604 && e <= 8607 || e === 8608 || e >= 8609 && e <= 8610 || e === 8611 || e >= 8612 && e <= 8613 || e === 8614 || e >= 8615 && e <= 8621 || e === 8622 || e >= 8623 && e <= 8653 || e >= 8654 && e <= 8655 || e >= 8656 && e <= 8657 || e === 8658 || e === 8659 || e === 8660 || e >= 8661 && e <= 8691 || e >= 8692 && e <= 8959 || e >= 8960 && e <= 8967 || e === 8968 || e === 8969 || e === 8970 || e === 8971 || e >= 8972 && e <= 8991 || e >= 8992 && e <= 8993 || e >= 8994 && e <= 9e3 || e === 9001 || e === 9002 || e >= 9003 && e <= 9083 || e === 9084 || e >= 9085 && e <= 9114 || e >= 9115 && e <= 9139 || e >= 9140 && e <= 9179 || e >= 9180 && e <= 9185 || e >= 9186 && e <= 9254 || e >= 9255 && e <= 9279 || e >= 9280 && e <= 9290 || e >= 9291 && e <= 9311 || e >= 9472 && e <= 9654 || e === 9655 || e >= 9656 && e <= 9664 || e === 9665 || e >= 9666 && e <= 9719 || e >= 9720 && e <= 9727 || e >= 9728 && e <= 9838 || e === 9839 || e >= 9840 && e <= 10087 || e === 10088 || e === 10089 || e === 10090 || e === 10091 || e === 10092 || e === 10093 || e === 10094 || e === 10095 || e === 10096 || e === 10097 || e === 10098 || e === 10099 || e === 10100 || e === 10101 || e >= 10132 && e <= 10175 || e >= 10176 && e <= 10180 || e === 10181 || e === 10182 || e >= 10183 && e <= 10213 || e === 10214 || e === 10215 || e === 10216 || e === 10217 || e === 10218 || e === 10219 || e === 10220 || e === 10221 || e === 10222 || e === 10223 || e >= 10224 && e <= 10239 || e >= 10240 && e <= 10495 || e >= 10496 && e <= 10626 || e === 10627 || e === 10628 || e === 10629 || e === 10630 || e === 10631 || e === 10632 || e === 10633 || e === 10634 || e === 10635 || e === 10636 || e === 10637 || e === 10638 || e === 10639 || e === 10640 || e === 10641 || e === 10642 || e === 10643 || e === 10644 || e === 10645 || e === 10646 || e === 10647 || e === 10648 || e >= 10649 && e <= 10711 || e === 10712 || e === 10713 || e === 10714 || e === 10715 || e >= 10716 && e <= 10747 || e === 10748 || e === 10749 || e >= 10750 && e <= 11007 || e >= 11008 && e <= 11055 || e >= 11056 && e <= 11076 || e >= 11077 && e <= 11078 || e >= 11079 && e <= 11084 || e >= 11085 && e <= 11123 || e >= 11124 && e <= 11125 || e >= 11126 && e <= 11157 || e === 11158 || e >= 11159 && e <= 11263 || e >= 11776 && e <= 11777 || e === 11778 || e === 11779 || e === 11780 || e === 11781 || e >= 11782 && e <= 11784 || e === 11785 || e === 11786 || e === 11787 || e === 11788 || e === 11789 || e >= 11790 && e <= 11798 || e === 11799 || e >= 11800 && e <= 11801 || e === 11802 || e === 11803 || e === 11804 || e === 11805 || e >= 11806 && e <= 11807 || e === 11808 || e === 11809 || e === 11810 || e === 11811 || e === 11812 || e === 11813 || e === 11814 || e === 11815 || e === 11816 || e === 11817 || e >= 11818 && e <= 11822 || e === 11823 || e >= 11824 && e <= 11833 || e >= 11834 && e <= 11835 || e >= 11836 && e <= 11839 || e === 11840 || e === 11841 || e === 11842 || e >= 11843 && e <= 11855 || e >= 11856 && e <= 11857 || e === 11858 || e >= 11859 && e <= 11903 || e >= 12289 && e <= 12291 || e === 12296 || e === 12297 || e === 12298 || e === 12299 || e === 12300 || e === 12301 || e === 12302 || e === 12303 || e === 12304 || e === 12305 || e >= 12306 && e <= 12307 || e === 12308 || e === 12309 || e === 12310 || e === 12311 || e === 12312 || e === 12313 || e === 12314 || e === 12315 || e === 12316 || e === 12317 || e >= 12318 && e <= 12319 || e === 12320 || e === 12336 || e === 64830 || e === 64831 || e >= 65093 && e <= 65094;
}
function Xr(e) {
  e.forEach(function(t) {
    if (delete t.location, ui(t) || fi(t))
      for (var r in t.options)
        delete t.options[r].location, Xr(t.options[r].value);
    else si(t) && hi(t.style) || (oi(t) || li(t)) && Fr(t.style) ? delete t.style.location : ci(t) && Xr(t.children);
  });
}
function Vs(e, t) {
  t === void 0 && (t = {}), t = F({ shouldParseSkeletons: !0, requiresOtherClause: !0 }, t);
  var r = new Us(e, t).parse();
  if (r.err) {
    var n = SyntaxError(R[r.err.kind]);
    throw n.location = r.err.location, n.originalMessage = r.err.message, n;
  }
  return t?.captureLocation || Xr(r.val), r.val;
}
var Tt;
(function(e) {
  e.MISSING_VALUE = "MISSING_VALUE", e.INVALID_VALUE = "INVALID_VALUE", e.MISSING_INTL_API = "MISSING_INTL_API";
})(Tt || (Tt = {}));
var cr = (
  /** @class */
  (function(e) {
    fr(t, e);
    function t(r, n, i) {
      var a = e.call(this, r) || this;
      return a.code = n, a.originalMessage = i, a;
    }
    return t.prototype.toString = function() {
      return "[formatjs Error: ".concat(this.code, "] ").concat(this.message);
    }, t;
  })(Error)
), In = (
  /** @class */
  (function(e) {
    fr(t, e);
    function t(r, n, i, a) {
      return e.call(this, 'Invalid values for "'.concat(r, '": "').concat(n, '". Options are "').concat(Object.keys(i).join('", "'), '"'), Tt.INVALID_VALUE, a) || this;
    }
    return t;
  })(cr)
), zs = (
  /** @class */
  (function(e) {
    fr(t, e);
    function t(r, n, i) {
      return e.call(this, 'Value for "'.concat(r, '" must be of type ').concat(n), Tt.INVALID_VALUE, i) || this;
    }
    return t;
  })(cr)
), Xs = (
  /** @class */
  (function(e) {
    fr(t, e);
    function t(r, n) {
      return e.call(this, 'The intl string context variable "'.concat(r, '" was not provided to the string "').concat(n, '"'), Tt.MISSING_VALUE, n) || this;
    }
    return t;
  })(cr)
), be;
(function(e) {
  e[e.literal = 0] = "literal", e[e.object = 1] = "object";
})(be || (be = {}));
function qs(e) {
  return e.length < 2 ? e : e.reduce(function(t, r) {
    var n = t[t.length - 1];
    return !n || n.type !== be.literal || r.type !== be.literal ? t.push(r) : n.value += r.value, t;
  }, []);
}
function Ws(e) {
  return typeof e == "function";
}
function Kt(e, t, r, n, i, a, o) {
  if (e.length === 1 && En(e[0]))
    return [
      {
        type: be.literal,
        value: e[0].value
      }
    ];
  for (var s = [], c = 0, f = e; c < f.length; c++) {
    var h = f[c];
    if (En(h)) {
      s.push({
        type: be.literal,
        value: h.value
      });
      continue;
    }
    if (ms(h)) {
      typeof a == "number" && s.push({
        type: be.literal,
        value: r.getNumberFormat(t).format(a)
      });
      continue;
    }
    var v = h.value;
    if (!(i && v in i))
      throw new Xs(v, o);
    var m = i[v];
    if (ps(h)) {
      (!m || typeof m == "string" || typeof m == "number") && (m = typeof m == "string" || typeof m == "number" ? String(m) : ""), s.push({
        type: typeof m == "string" ? be.literal : be.object,
        value: m
      });
      continue;
    }
    if (oi(h)) {
      var y = typeof h.style == "string" ? n.date[h.style] : Fr(h.style) ? h.style.parsedOptions : void 0;
      s.push({
        type: be.literal,
        value: r.getDateTimeFormat(t, y).format(m)
      });
      continue;
    }
    if (li(h)) {
      var y = typeof h.style == "string" ? n.time[h.style] : Fr(h.style) ? h.style.parsedOptions : n.time.medium;
      s.push({
        type: be.literal,
        value: r.getDateTimeFormat(t, y).format(m)
      });
      continue;
    }
    if (si(h)) {
      var y = typeof h.style == "string" ? n.number[h.style] : hi(h.style) ? h.style.parsedOptions : void 0;
      y && y.scale && (m = m * (y.scale || 1)), s.push({
        type: be.literal,
        value: r.getNumberFormat(t, y).format(m)
      });
      continue;
    }
    if (ci(h)) {
      var _ = h.children, S = h.value, B = i[S];
      if (!Ws(B))
        throw new zs(S, "function", o);
      var p = Kt(_, t, r, n, i, a), g = B(p.map(function(x) {
        return x.value;
      }));
      Array.isArray(g) || (g = [g]), s.push.apply(s, g.map(function(x) {
        return {
          type: typeof x == "string" ? be.literal : be.object,
          value: x
        };
      }));
    }
    if (ui(h)) {
      var T = h.options[m] || h.options.other;
      if (!T)
        throw new In(h.value, m, Object.keys(h.options), o);
      s.push.apply(s, Kt(T.value, t, r, n, i));
      continue;
    }
    if (fi(h)) {
      var T = h.options["=".concat(m)];
      if (!T) {
        if (!Intl.PluralRules)
          throw new cr(`Intl.PluralRules is not available in this environment.
Try polyfilling it using "@formatjs/intl-pluralrules"
`, Tt.MISSING_INTL_API, o);
        var E = r.getPluralRules(t, { type: h.pluralType }).select(m - (h.offset || 0));
        T = h.options[E] || h.options.other;
      }
      if (!T)
        throw new In(h.value, m, Object.keys(h.options), o);
      s.push.apply(s, Kt(T.value, t, r, n, i, m - (h.offset || 0)));
      continue;
    }
  }
  return qs(s);
}
function Zs(e, t) {
  return t ? F(F(F({}, e || {}), t || {}), Object.keys(e).reduce(function(r, n) {
    return r[n] = F(F({}, e[n]), t[n] || {}), r;
  }, {})) : e;
}
function Ys(e, t) {
  return t ? Object.keys(e).reduce(function(r, n) {
    return r[n] = Zs(e[n], t[n]), r;
  }, F({}, e)) : e;
}
function Pr(e) {
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
function Js(e) {
  return e === void 0 && (e = {
    number: {},
    dateTime: {},
    pluralRules: {}
  }), {
    getNumberFormat: Sr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.NumberFormat).bind.apply(t, Tr([void 0], r, !1)))();
    }, {
      cache: Pr(e.number),
      strategy: Ar.variadic
    }),
    getDateTimeFormat: Sr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.DateTimeFormat).bind.apply(t, Tr([void 0], r, !1)))();
    }, {
      cache: Pr(e.dateTime),
      strategy: Ar.variadic
    }),
    getPluralRules: Sr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.PluralRules).bind.apply(t, Tr([void 0], r, !1)))();
    }, {
      cache: Pr(e.pluralRules),
      strategy: Ar.variadic
    })
  };
}
var Qs = (
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
        var h = f.reduce(function(v, m) {
          return !v.length || m.type !== be.literal || typeof v[v.length - 1] != "string" ? v.push(m.value) : v[v.length - 1] += m.value, v;
        }, []);
        return h.length <= 1 ? h[0] || "" : h;
      }, this.formatToParts = function(c) {
        return Kt(a.ast, a.locales, a.formatters, a.formats, c, void 0, a.message);
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
        var s = ss(o, ["formatters"]);
        this.ast = e.__parse(t, F(F({}, s), { locale: this.resolvedLocale }));
      } else
        this.ast = t;
      if (!Array.isArray(this.ast))
        throw new TypeError("A message must be provided as a String or AST.");
      this.formats = Ys(e.formats, n), this.formatters = i && i.formatters || Js(this.formatterCache);
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
    }, e.__parse = Vs, e.formats = {
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
function Ks(e, t) {
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
const $e = {}, $s = (e, t, r) => r && (t in $e || ($e[t] = {}), e in $e[t] || ($e[t][e] = r), r), yi = (e, t) => {
  if (t == null)
    return;
  if (t in $e && e in $e[t])
    return $e[t][e];
  const r = hr(t);
  for (let n = 0; n < r.length; n++) {
    const i = r[n], a = to(i, e);
    if (a)
      return $s(e, t, a);
  }
};
let nn;
const zt = Vt({});
function eo(e) {
  return nn[e] || null;
}
function xi(e) {
  return e in nn;
}
function to(e, t) {
  if (!xi(e))
    return null;
  const r = eo(e);
  return Ks(r, t);
}
function ro(e) {
  if (e == null)
    return;
  const t = hr(e);
  for (let r = 0; r < t.length; r++) {
    const n = t[r];
    if (xi(n))
      return n;
  }
}
function no(e, ...t) {
  delete $e[e], zt.update((r) => (r[e] = as.all([r[e] || {}, ...t]), r));
}
At(
  [zt],
  ([e]) => Object.keys(e)
);
zt.subscribe((e) => nn = e);
const $t = {};
function io(e, t) {
  $t[e].delete(t), $t[e].size === 0 && delete $t[e];
}
function Ei(e) {
  return $t[e];
}
function ao(e) {
  return hr(e).map((t) => {
    const r = Ei(t);
    return [t, r ? [...r] : []];
  }).filter(([, t]) => t.length > 0);
}
function qr(e) {
  return e == null ? !1 : hr(e).some(
    (t) => {
      var r;
      return (r = Ei(t)) == null ? void 0 : r.size;
    }
  );
}
function so(e, t) {
  return Promise.all(
    t.map((n) => (io(e, n), n().then((i) => i.default || i)))
  ).then((n) => no(e, ...n));
}
const Rt = {};
function wi(e) {
  if (!qr(e))
    return e in Rt ? Rt[e] : Promise.resolve();
  const t = ao(e);
  return Rt[e] = Promise.all(
    t.map(
      ([r, n]) => so(r, n)
    )
  ).then(() => {
    if (qr(e))
      return wi(e);
    delete Rt[e];
  }), Rt[e];
}
const oo = {
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
}, lo = {
  fallbackLocale: null,
  loadingDelay: 200,
  formats: oo,
  warnOnMissingMessages: !0,
  handleMissingMessage: void 0,
  ignoreTag: !0
}, uo = lo;
function St() {
  return uo;
}
const Ir = Vt(!1);
var fo = Object.defineProperty, co = Object.defineProperties, ho = Object.getOwnPropertyDescriptors, Bn = Object.getOwnPropertySymbols, po = Object.prototype.hasOwnProperty, mo = Object.prototype.propertyIsEnumerable, On = (e, t, r) => t in e ? fo(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, vo = (e, t) => {
  for (var r in t || (t = {}))
    po.call(t, r) && On(e, r, t[r]);
  if (Bn)
    for (var r of Bn(t))
      mo.call(t, r) && On(e, r, t[r]);
  return e;
}, go = (e, t) => co(e, ho(t));
let Wr;
const nr = Vt(null);
function Mn(e) {
  return e.split("-").map((t, r, n) => n.slice(0, r + 1).join("-")).reverse();
}
function hr(e, t = St().fallbackLocale) {
  const r = Mn(e);
  return t ? [.../* @__PURE__ */ new Set([...r, ...Mn(t)])] : r;
}
function ft() {
  return Wr ?? void 0;
}
nr.subscribe((e) => {
  Wr = e ?? void 0, typeof window < "u" && e != null && document.documentElement.setAttribute("lang", e);
});
const bo = (e) => {
  if (e && ro(e) && qr(e)) {
    const { loadingDelay: t } = St();
    let r;
    return typeof window < "u" && ft() != null && t ? r = window.setTimeout(
      () => Ir.set(!0),
      t
    ) : Ir.set(!0), wi(e).then(() => {
      nr.set(e);
    }).finally(() => {
      clearTimeout(r), Ir.set(!1);
    });
  }
  return nr.set(e);
}, Ht = go(vo({}, nr), {
  set: bo
}), dr = (e) => {
  const t = /* @__PURE__ */ Object.create(null);
  return (n) => {
    const i = JSON.stringify(n);
    return i in t ? t[i] : t[i] = e(n);
  };
};
var _o = Object.defineProperty, ir = Object.getOwnPropertySymbols, Ti = Object.prototype.hasOwnProperty, Si = Object.prototype.propertyIsEnumerable, Ln = (e, t, r) => t in e ? _o(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, an = (e, t) => {
  for (var r in t || (t = {}))
    Ti.call(t, r) && Ln(e, r, t[r]);
  if (ir)
    for (var r of ir(t))
      Si.call(t, r) && Ln(e, r, t[r]);
  return e;
}, Pt = (e, t) => {
  var r = {};
  for (var n in e)
    Ti.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && ir)
    for (var n of ir(e))
      t.indexOf(n) < 0 && Si.call(e, n) && (r[n] = e[n]);
  return r;
};
const Ft = (e, t) => {
  const { formats: r } = St();
  if (e in r && t in r[e])
    return r[e][t];
  throw new Error(`[svelte-i18n] Unknown "${t}" ${e} format.`);
}, yo = dr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = Pt(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format numbers');
    return n && (i = Ft("number", n)), new Intl.NumberFormat(r, i);
  }
), xo = dr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = Pt(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format dates');
    return n ? i = Ft("date", n) : Object.keys(i).length === 0 && (i = Ft("date", "short")), new Intl.DateTimeFormat(r, i);
  }
), Eo = dr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = Pt(t, ["locale", "format"]);
    if (r == null)
      throw new Error(
        '[svelte-i18n] A "locale" must be set to format time values'
      );
    return n ? i = Ft("time", n) : Object.keys(i).length === 0 && (i = Ft("time", "short")), new Intl.DateTimeFormat(r, i);
  }
), wo = (e = {}) => {
  var t = e, {
    locale: r = ft()
  } = t, n = Pt(t, [
    "locale"
  ]);
  return yo(an({ locale: r }, n));
}, To = (e = {}) => {
  var t = e, {
    locale: r = ft()
  } = t, n = Pt(t, [
    "locale"
  ]);
  return xo(an({ locale: r }, n));
}, So = (e = {}) => {
  var t = e, {
    locale: r = ft()
  } = t, n = Pt(t, [
    "locale"
  ]);
  return Eo(an({ locale: r }, n));
}, Ao = dr(
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  (e, t = ft()) => new Qs(e, t, St().formats, {
    ignoreTag: St().ignoreTag
  })
), Ho = (e, t = {}) => {
  var r, n, i, a;
  let o = t;
  typeof e == "object" && (o = e, e = o.id);
  const {
    values: s,
    locale: c = ft(),
    default: f
  } = o;
  if (c == null)
    throw new Error(
      "[svelte-i18n] Cannot format a message without first setting the initial locale."
    );
  let h = yi(e, c);
  if (!h)
    h = (a = (i = (n = (r = St()).handleMissingMessage) == null ? void 0 : n.call(r, { locale: c, id: e, defaultValue: f })) != null ? i : f) != null ? a : e;
  else if (typeof h != "string")
    return console.warn(
      `[svelte-i18n] Message with id "${e}" must be of type "string", found: "${typeof h}". Gettin its value through the "$format" method is deprecated; use the "json" method instead.`
    ), h;
  if (!s)
    return h;
  let v = h;
  try {
    v = Ao(h, c).format(s);
  } catch (m) {
    m instanceof Error && console.warn(
      `[svelte-i18n] Message "${e}" has syntax error:`,
      m.message
    );
  }
  return v;
}, Po = (e, t) => So(t).format(e), Io = (e, t) => To(t).format(e), Bo = (e, t) => wo(t).format(e), Oo = (e, t = ft()) => yi(e, t);
At([Ht, zt], () => Ho);
At([Ht], () => Po);
At([Ht], () => Io);
At([Ht], () => Bo);
At([Ht, zt], () => Oo);
const Mo = "__i18n__", Lo = [
  "label",
  "info",
  "placeholder",
  "description",
  "title",
  "value"
], No = [
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
function Co(e) {
  return typeof e == "string" && e.includes(Mo);
}
class Ro {
  load_component;
  #t = q(Ut({}));
  get shared() {
    return l(this.#t);
  }
  set shared(t) {
    w(this.#t, t, !0);
  }
  #r = q(Ut({}));
  get props() {
    return l(this.#r);
  }
  set props(t) {
    w(this.#r, t, !0);
  }
  #e = q((t) => t);
  get i18n() {
    return l(this.#e);
  }
  set i18n(t) {
    w(this.#e, t, !0);
  }
  translatable_props = {};
  dispatcher;
  last_update = null;
  shared_props = No;
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
    for (const n of Lo)
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
    ), Pe(() => {
      for (const n in t.shared_props)
        this._is_i18n_managed(`shared.${n}`, t.shared_props[n]) || (this.shared[n] = t.shared_props[n]);
      for (const n in t.props)
        this._is_i18n_managed(`props.${n}`, t.props[n]) || (this.props[n] = t.props[n]);
      this.register_component(
        t.shared_props.id,
        // @ts-ignore
        this.set_data.bind(this),
        this.get_data.bind(this)
      ), ue(() => {
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
    return ba(this.props);
  }
  update(t) {
    this.set_data(t);
  }
  set_data(t) {
    for (const r in t) {
      const n = t[r], i = Co(n) ? this._translate_and_store(this.shared_props.includes(r) ? "shared" : "props", r, n) : n;
      if (this.shared_props.includes(r)) {
        const a = r;
        this.shared[a] = i;
        continue;
      }
      this.props[r] = i;
    }
  }
  watch_for_change() {
    Pe(() => {
      this.mounted || (this.old_value = this.props.value, this.mounted = !0), this.old_value != this.props.value && (this.old_value = this.props.value, this.dispatch("change"));
    });
  }
}
da();
var Do = /* @__PURE__ */ Qn('<svg class="resize-handle svelte-1stq1b1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><line x1="1" y1="9" x2="9" y2="1" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line><line x1="5" y1="9" x2="9" y2="5" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line></svg>'), Nn = /* @__PURE__ */ pe("<!> <!>", 1), ko = /* @__PURE__ */ pe('<div class="placeholder svelte-1stq1b1"></div>');
function Uo(e, t) {
  lr(t, !1);
  let r = O(t, "height", 8, void 0), n = O(t, "min_height", 8, void 0), i = O(t, "max_height", 8, void 0), a = O(t, "width", 8, void 0), o = O(t, "elem_id", 8, ""), s = O(t, "elem_classes", 24, () => []), c = O(t, "variant", 8, "solid"), f = O(t, "border_mode", 8, "base"), h = O(t, "padding", 8, !0), v = O(t, "type", 8, "normal"), m = O(t, "test_id", 8, void 0), y = O(t, "explicit_call", 8, !1), _ = O(t, "container", 8, !0), S = O(t, "visible", 8, !0), B = O(t, "allow_overflow", 8, !0), p = O(t, "overflow_behavior", 8, "auto"), g = O(t, "scale", 8, null), T = O(t, "min_width", 8, 0), E = O(t, "flex", 12, !1), x = O(t, "resizable", 8, !1), H = O(t, "rtl", 8, !1), P = O(t, "fullscreen", 12, !1), L = O(t, "label", 8, void 0), U = st(P()), G = st(), he = v() === "fieldset" ? "fieldset" : "div", xe = st(0), fe = st(0), X = st(null);
  function we(ne) {
    P() && ne.key === "Escape" && P(!1);
  }
  const _e = (ne) => {
    if (ne !== void 0) {
      if (typeof ne == "number")
        return ne + "px";
      if (typeof ne == "string")
        return ne;
    }
  }, qe = (ne) => {
    let Be = ne.clientY;
    const le = (j) => {
      const Z = j.clientY - Be;
      Be = j.clientY, ma(G, l(G).style.height = `${l(G).offsetHeight + Z}px`);
    }, Ee = () => {
      window.removeEventListener("mousemove", le), window.removeEventListener("mouseup", Ee);
    };
    window.addEventListener("mousemove", le), window.addEventListener("mouseup", Ee);
  };
  fn(
    () => (He(P()), l(U), l(G)),
    () => {
      P() !== l(U) && (w(U, P()), P() ? (w(X, l(G).getBoundingClientRect()), w(xe, l(G).offsetHeight), w(fe, l(G).offsetWidth), window.addEventListener("keydown", we)) : (w(X, null), window.removeEventListener("keydown", we)));
    }
  ), fn(() => He(S()), () => {
    S() || E(!1);
  }), pa(), Ja();
  var Ie = _t(), Re = ge(Ie);
  {
    var rt = (ne) => {
      var Be = Nn(), le = ge(Be);
      Da(le, () => he, !1, (Z, We) => {
        rr(Z, (me) => w(G, me), () => l(G)), Ya(
          Z,
          (me, Ge) => ({
            "data-testid": m(),
            id: o(),
            class: `block ${me ?? ""}`,
            dir: H() ? "rtl" : "ltr",
            "aria-label": L(),
            style: "",
            [kt]: {
              hidden: S() === "hidden",
              padded: h(),
              flex: E(),
              border_focus: f() === "focus",
              border_contrast: f() === "contrast",
              "hide-container": !y() && !_(),
              fullscreen: P(),
              animating: P() && l(X) !== null,
              "auto-margin": g() === null
            },
            [bt]: Ge
          }),
          [
            () => (He(s()), ue(() => s()?.join(" ") || "")),
            () => ({
              height: (He(P()), He(r()), ue(() => P() ? void 0 : _e(r()))),
              "min-height": (He(P()), He(n()), ue(() => P() ? void 0 : _e(n()))),
              "max-height": (He(P()), He(i()), ue(() => P() ? void 0 : _e(i()))),
              "--start-top": (l(X), ue(() => l(X) ? `${l(X).top}px` : "0px")),
              "--start-left": (l(X), ue(() => l(X) ? `${l(X).left}px` : "0px")),
              "--start-width": (l(X), ue(() => l(X) ? `${l(X).width}px` : "0px")),
              "--start-height": (l(X), ue(() => l(X) ? `${l(X).height}px` : "0px")),
              width: (He(P()), He(a()), ue(() => P() ? void 0 : typeof a() == "number" ? `calc(min(${a()}px, 100%))` : _e(a()))),
              "border-style": c(),
              overflow: B() ? p() : "hidden",
              "flex-grow": g(),
              "min-width": `calc(min(${T()}px, 100%))`
            })
          ],
          void 0,
          void 0,
          "svelte-1stq1b1"
        );
        var ae = Nn(), De = ge(ae);
        Dr(De, t, "default", {});
        var Fe = J(De, 2);
        {
          var ct = (me) => {
            var Ge = Do();
            gt("mousedown", Ge, qe), D(me, Ge);
          };
          ee(Fe, (me) => {
            x() && me(ct);
          });
        }
        D(We, ae);
      });
      var Ee = J(le, 2);
      {
        var j = (Z) => {
          var We = ko();
          let ae;
          re(() => ae = Ce(We, "", ae, {
            height: l(xe) + "px",
            width: l(fe) + "px"
          })), D(Z, We);
        };
        ee(Ee, (Z) => {
          P() && Z(j);
        });
      }
      D(ne, Be);
    };
    ee(Re, (ne) => {
      (S() === !0 || S() === "hidden") && ne(rt);
    });
  }
  D(e, Ie), or();
}
var Fo = /* @__PURE__ */ pe('<span class="svelte-vvirtv"> </span>'), Go = /* @__PURE__ */ pe("<button><!> <div><!> <!></div></button>");
function Cn(e, t) {
  let r = O(t, "label", 3, ""), n = O(t, "show_label", 3, !1), i = O(t, "pending", 3, !1), a = O(t, "size", 3, "small"), o = O(t, "padded", 3, !0), s = O(t, "highlight", 3, !1), c = O(t, "disabled", 3, !1), f = O(t, "hasPopup", 3, !1), h = O(t, "color", 3, "var(--block-label-text-color)"), v = O(t, "transparent", 3, !1), m = O(t, "background", 3, "var(--block-background-fill)"), y = O(t, "border", 3, "transparent"), _ = Le(() => s() ? "var(--color-accent)" : h());
  var S = Go();
  let B, p;
  var g = oe(S);
  {
    var T = (U) => {
      var G = Fo(), he = oe(G);
      re(() => ve(he, r())), D(U, G);
    };
    ee(g, (U) => {
      n() && U(T);
    });
  }
  var E = J(g, 2);
  let x;
  var H = oe(E);
  Na(H, () => t.Icon, (U, G) => {
    G(U, {});
  });
  var P = J(H, 2);
  {
    var L = (U) => {
      var G = _t(), he = ge(G);
      Ha(he, () => t.children), D(U, G);
    };
    ee(P, (U) => {
      t.children && U(L);
    });
  }
  re(() => {
    B = tt(S, 1, "icon-button svelte-vvirtv", null, B, {
      pending: i(),
      padded: o(),
      highlight: s(),
      transparent: v()
    }), S.disabled = c(), xt(S, "aria-label", r()), xt(S, "aria-haspopup", f()), xt(S, "title", r()), p = Ce(S, "", p, {
      "--border-color": y(),
      color: !c() && l(_) ? l(_) : "var(--block-label-text-color)",
      "--bg-color": c() ? "auto" : m()
    }), x = tt(E, 1, "svelte-vvirtv", null, x, {
      "x-small": a() === "x-small",
      small: a() === "small",
      large: a() === "large",
      medium: a() === "medium"
    });
  }), Ke("click", S, function(...U) {
    t.onclick?.apply(this, U);
  }), D(e, S);
}
jt(["click"]);
var jo = /* @__PURE__ */ Qn('<svg width="100%" height="100%" viewBox="0 0 24 24" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:space="preserve" stroke="currentColor" style="fill-rule:evenodd;clip-rule:evenodd;stroke-linecap:round;stroke-linejoin:round;"><g transform="matrix(1.14096,-0.140958,-0.140958,1.14096,-0.0559523,0.0559523)"><path d="M18,6L6.087,17.913" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></g><path d="M4.364,4.364L19.636,19.636" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></svg>');
function Rn(e) {
  var t = jo();
  D(e, t);
}
jt(["click"]);
function Br(e) {
  let t = ["", "k", "M", "G", "T", "P", "E", "Z"], r = 0;
  for (; e > 1e3 && r < t.length - 1; )
    e /= 1e3, r++;
  let n = t[r];
  return (Number.isInteger(e) ? e : e.toFixed(1)) + n;
}
function Dn(e) {
  return Object.prototype.toString.call(e) === "[object Date]";
}
function Zr(e, t, r, n) {
  if (typeof r == "number" || Dn(r)) {
    const i = n - r, a = (r - t) / (e.dt || 1 / 60), o = e.opts.stiffness * i, s = e.opts.damping * a, c = (o - s) * e.inv_mass, f = (a + c) * e.dt;
    return Math.abs(f) < e.opts.precision && Math.abs(i) < e.opts.precision ? n : (e.settled = !1, Dn(r) ? new Date(r.getTime() + f) : r + f);
  } else {
    if (Array.isArray(r))
      return r.map(
        (i, a) => (
          // @ts-ignore
          Zr(e, t[a], r[a], n[a])
        )
      );
    if (typeof r == "object") {
      const i = {};
      for (const a in r)
        i[a] = Zr(e, t[a], r[a], n[a]);
      return i;
    } else
      throw new Error(`Cannot spring ${typeof r} values`);
  }
}
function kn(e, t = {}) {
  const r = Vt(e), { stiffness: n = 0.15, damping: i = 0.8, precision: a = 0.01 } = t;
  let o, s, c, f = (
    /** @type {T} */
    e
  ), h = (
    /** @type {T | undefined} */
    e
  ), v = 1, m = 0, y = !1;
  function _(B, p = {}) {
    h = B;
    const g = c = {};
    return e == null || p.hard || S.stiffness >= 1 && S.damping >= 1 ? (y = !0, o = Ne.now(), f = B, r.set(e = h), Promise.resolve()) : (p.soft && (m = 1 / ((p.soft === !0 ? 0.5 : +p.soft) * 60), v = 0), s || (o = Ne.now(), y = !1, s = Ra((T) => {
      if (y)
        return y = !1, s = null, !1;
      v = Math.min(v + m, 1);
      const E = Math.min(T - o, 1e3 / 30), x = {
        inv_mass: v,
        opts: S,
        settled: !0,
        dt: E * 60 / 1e3
      }, H = Zr(x, f, e, h);
      return o = T, f = /** @type {T} */
      e, r.set(e = /** @type {T} */
      H), x.settled && (s = null), !x.settled;
    })), new Promise((T) => {
      s.promise.then(() => {
        g === c && T();
      });
    }));
  }
  const S = {
    set: _,
    update: (B, p) => _(B(
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
  return S;
}
var Vo = /* @__PURE__ */ pe('<div><svg viewBox="-1200 -1200 3000 3000" fill="none" xmlns="http://www.w3.org/2000/svg" class="svelte-m6d381"><g><path d="M255.926 0.754768L509.702 139.936V221.027L255.926 81.8465V0.754768Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M509.69 139.936L254.981 279.641V361.255L509.69 221.55V139.936Z" fill="#FF7C00" class="svelte-m6d381"></path><path d="M0.250138 139.937L254.981 279.641V361.255L0.250138 221.55V139.937Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M255.923 0.232622L0.236328 139.936V221.55L255.923 81.8469V0.232622Z" fill="#FF7C00" class="svelte-m6d381"></path></g><g><path d="M255.926 141.5L509.702 280.681V361.773L255.926 222.592V141.5Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M509.69 280.679L254.981 420.384V501.998L509.69 362.293V280.679Z" fill="#FF7C00" class="svelte-m6d381"></path><path d="M0.250138 280.681L254.981 420.386V502L0.250138 362.295V280.681Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M255.923 140.977L0.236328 280.68V362.294L255.923 222.591V140.977Z" fill="#FF7C00" class="svelte-m6d381"></path></g></svg></div>');
function zo(e, t) {
  lr(t, !0);
  const r = () => cn(c, "$top", i), n = () => cn(f, "$bottom", i), [i, a] = xa();
  var o = this && this.__awaiter || function(T, E, x, H) {
    function P(L) {
      return L instanceof x ? L : new x(function(U) {
        U(L);
      });
    }
    return new (x || (x = Promise))(function(L, U) {
      function G(fe) {
        try {
          xe(H.next(fe));
        } catch (X) {
          U(X);
        }
      }
      function he(fe) {
        try {
          xe(H.throw(fe));
        } catch (X) {
          U(X);
        }
      }
      function xe(fe) {
        fe.done ? L(fe.value) : P(fe.value).then(G, he);
      }
      xe((H = H.apply(T, E || [])).next());
    });
  };
  let s = O(t, "margin", 3, !0);
  const c = kn([0, 0]), f = kn([0, 0]);
  let h = q(!1);
  function v() {
    return o(this, void 0, void 0, function* () {
      yield Promise.all([c.set([125, 140]), f.set([-125, -140])]), yield Promise.all([c.set([-125, 140]), f.set([125, -140])]), yield Promise.all([c.set([-125, 0]), f.set([125, -0])]), yield Promise.all([c.set([125, 0]), f.set([-125, 0])]);
    });
  }
  function m() {
    return o(this, void 0, void 0, function* () {
      yield v(), l(h) || m();
    });
  }
  function y() {
    return o(this, void 0, void 0, function* () {
      yield Promise.all([c.set([125, 0]), f.set([-125, 0])]), m();
    });
  }
  Pe(() => (y(), () => {
    w(h, !0);
  }));
  var _ = Vo();
  let S;
  var B = oe(_), p = oe(B), g = J(p);
  re(() => {
    S = tt(_, 1, "svelte-m6d381", null, S, { margin: s() }), Ce(p, `transform: translate(${r()[0] ?? ""}px, ${r()[1] ?? ""}px);`), Ce(g, `transform: translate(${n()[0] ?? ""}px, ${n()[1] ?? ""}px);`);
  }), D(e, _), or(), a();
}
var Xo = function(e, t, r, n) {
  function i(a) {
    return a instanceof r ? a : new r(function(o) {
      o(a);
    });
  }
  return new (r || (r = Promise))(function(a, o) {
    function s(h) {
      try {
        f(n.next(h));
      } catch (v) {
        o(v);
      }
    }
    function c(h) {
      try {
        f(n.throw(h));
      } catch (v) {
        o(v);
      }
    }
    function f(h) {
      h.done ? a(h.value) : i(h.value).then(s, c);
    }
    f((n = n.apply(e, t || [])).next());
  });
};
let Jt = [], Or = !1;
const qo = typeof window < "u", Ai = qo ? window.requestAnimationFrame : (e) => {
};
function Wo(e) {
  return Xo(this, arguments, void 0, function* (t, r = !0) {
    if (!(window.__gradio_mode__ === "website" || window.__gradio_mode__ !== "app" && r !== !0)) {
      if (Jt.push(t), !Or) Or = !0;
      else return;
      yield va(), Ai(() => {
        let n = [0, 0];
        for (let i = 0; i < Jt.length; i++) {
          const o = Jt[i].getBoundingClientRect();
          (i === 0 || o.top + window.scrollY <= n[0]) && (n[0] = o.top + window.scrollY, n[1] = i);
        }
        window.scrollTo({ top: n[0] - 20, behavior: "smooth" }), Or = !1, Jt = [];
      });
    }
  });
}
var Zo = /* @__PURE__ */ pe('<div class="validation-error svelte-124hqw6"> <button class="svelte-124hqw6"><!></button></div>'), Yo = /* @__PURE__ */ pe('<div class="eta-bar svelte-124hqw6"></div>'), Jo = /* @__PURE__ */ pe("<!> ", 1), Qo = /* @__PURE__ */ pe("<!> <!> <!> <!>", 1), Ko = /* @__PURE__ */ pe('<div class="progress-level svelte-124hqw6"><div class="progress-level-inner svelte-124hqw6"><!></div> <div class="progress-bar-wrap svelte-124hqw6"><div class="progress-bar svelte-124hqw6"></div></div></div>'), $o = /* @__PURE__ */ pe('<p class="loading svelte-124hqw6"> </p> <!>', 1), el = /* @__PURE__ */ pe("<!> <div><!> <!></div> <!> <!>", 1), tl = /* @__PURE__ */ pe('<div class="clear-status svelte-124hqw6"><!></div> <span class="error svelte-124hqw6"> </span> <!>', 1), rl = /* @__PURE__ */ pe("<div> <!> </div>"), nl = /* @__PURE__ */ pe('<div data-testid="status-tracker"><!> <!></div> <!>', 1);
function il(e, t) {
  lr(t, !0);
  let r = O(t, "eta", 3, null), n = O(t, "scroll_to_output", 3, !1), i = O(t, "timer", 3, !0), a = O(t, "show_progress", 3, "full"), o = O(t, "message", 3, null), s = O(t, "progress", 3, null), c = O(t, "variant", 3, "default"), f = O(t, "loading_text", 3, "Loading..."), h = O(t, "absolute", 3, !0), v = O(t, "translucent", 3, !1), m = O(t, "border", 3, !1), y = O(t, "validation_error", 7, null), _ = O(t, "show_validation_error", 3, !0), S = O(t, "type", 3, null), B = O(t, "used_cache", 3, null), p = O(t, "cache_duration", 3, null), g = O(t, "avg_time", 3, null), T, E = !1, x = q(0), H = q(null), P = q(null), L = q(!1), U = q(null), G = q(!1), he = q(!1), xe = q(null), fe = q(null), X = q("from cache"), we = q(!1), _e = null, qe = null;
  const Ie = Le(() => !(_() && y()) && (S() === "input" || !t.status || t.status === "complete" || a() === "hidden" || t.status == "streaming"));
  let Re = q(0);
  const rt = Le(() => l(P) === null || l(P) <= 0 || !l(Re) ? 0 : Math.min(l(Re) / l(P), 1)), ne = Le(() => l(Re).toFixed(1));
  let Be = Le(() => s() == null), le = Le(() => r() !== null && r() !== void 0 ? r() : l(H));
  function Ee() {
    Ai(() => {
      w(Re, (performance.now() - l(x)) / 1e3), E && Ee();
    });
  }
  let j = Le(() => {
    let z = null;
    s() != null ? z = s().map((ie) => {
      if (ie.index != null && ie.length != null)
        return ie.index / ie.length;
      if (ie.progress != null)
        return ie.progress;
    }) : z = null;
    let K, te = "";
    return z ? (K = z[z.length - 1], K === 0 ? te = "0" : te = "150ms") : K = void 0, {
      progress_level: z,
      last_progress_level: K,
      progress_bar_transition: te
    };
  });
  function Z() {
    E || (w(H, w(U, null), !0), w(x, performance.now(), !0), E = !0, Ee());
  }
  function We() {
    w(H, w(U, null), !0), E && (E = !1);
  }
  Pe(() => {
    t.status === "pending" ? Z() : ue(() => {
      We();
    });
  }), Pe(() => {
    T && n() && (t.status === "pending" || t.status === "complete") && Wo(T, t.autoscroll);
  }), Pe(() => {
    l(le) != null && l(H) !== l(le) && (w(P, (performance.now() - l(x)) / 1e3 + l(le)), w(U, l(P).toFixed(1), !0), w(H, l(le), !0));
  });
  function ae() {
    w(L, !1);
  }
  Pe(() => {
    ue(() => {
      ae();
    }), t.status === "error" && o() && w(L, !0);
  }), Pe(() => {
    t.status === "complete" && S() === "output" && B() && p() != null && (w(xe, p().toFixed(1), !0), w(X, B() === "full" ? "from cache" : "used cache", !0), w(we, g() != null && g() > p() && g() > 0, !0), w(fe, l(we) ? g().toFixed(1) : null, !0), w(G, !0), w(he, !1), _e && clearTimeout(_e), qe && clearTimeout(qe), _e = setTimeout(
      () => {
        w(he, !0), qe = setTimeout(
          () => {
            w(G, !1), w(he, !1);
          },
          500
        );
      },
      1750
    ));
  });
  var De = nl(), Fe = ge(De);
  let ct, me;
  var Ge = oe(Fe);
  {
    var Xt = (z) => {
      var K = Zo(), te = oe(K), ie = J(te), ye = oe(ie);
      {
        let Oe = Le(() => t.i18n ? t.i18n("common.clear") : "Clear");
        Cn(ye, {
          get Icon() {
            return Rn;
          },
          get label() {
            return l(Oe);
          },
          disabled: !1,
          size: "x-small",
          background: "var(--background-fill-primary)",
          color: "var(--error-background-text)",
          border: "var(--border-color-primary)",
          onclick: () => y(null)
        });
      }
      re(() => ve(te, `${y() ?? ""} `)), D(z, K);
    };
    ee(Ge, (z) => {
      y() && _() && z(Xt);
    });
  }
  var qt = J(Ge, 2);
  {
    var pr = (z) => {
      var K = el(), te = ge(K);
      {
        var ie = (V) => {
          var Y = Yo();
          let Te;
          re(() => Te = Ce(Y, "", Te, {
            transform: `translateX(${(l(rt) || 0) * 100 - 100}%)`
          })), D(V, Y);
        };
        ee(te, (V) => {
          c() === "default" && l(Be) && a() === "full" && V(ie);
        });
      }
      var ye = J(te, 2);
      let Oe;
      var Me = oe(ye);
      {
        var ke = (V) => {
          var Y = _t(), Te = ge(Y);
          pn(Te, 17, s, hn, (je, Se) => {
            var dt = _t(), de = ge(dt);
            {
              var pt = (Ve) => {
                var it = Jo(), mt = ge(it);
                {
                  var Lt = (Ae) => {
                    var ze = Ue();
                    re((at, u) => ve(ze, `${at ?? ""}/${u ?? ""}`), [
                      () => Br(l(Se).index || 0),
                      () => Br(l(Se).length)
                    ]), D(Ae, ze);
                  }, Ye = (Ae) => {
                    var ze = Ue();
                    re((at) => ve(ze, at), [() => Br(l(Se).index || 0)]), D(Ae, ze);
                  };
                  ee(mt, (Ae) => {
                    l(Se).length != null ? Ae(Lt) : Ae(Ye, -1);
                  });
                }
                var Je = J(mt);
                re(() => ve(Je, ` ${l(Se).unit ?? ""} |  `)), D(Ve, it);
              };
              ee(de, (Ve) => {
                l(Se).index != null && Ve(pt);
              });
            }
            D(je, dt);
          }), D(V, Y);
        }, Ze = (V) => {
          var Y = Ue();
          re(() => ve(Y, `queue: ${t.queue_position + 1}/${t.queue_size ?? ""} |`)), D(V, Y);
        }, nt = (V) => {
          var Y = Ue("processing |");
          D(V, Y);
        };
        ee(Me, (V) => {
          s() ? V(ke) : t.queue_position !== null && t.queue_size !== void 0 && t.queue_position >= 0 ? V(Ze, 1) : t.queue_position === 0 && V(nt, 2);
        });
      }
      var Bt = J(Me, 2);
      {
        var vr = (V) => {
          var Y = Ue();
          re(() => ve(Y, `${l(ne) ?? ""}${r() ? `/${l(U)}` : ""}s`)), D(V, Y);
        };
        ee(Bt, (V) => {
          i() && V(vr);
        });
      }
      var Wt = J(ye, 2);
      {
        var gr = (V) => {
          var Y = Ko(), Te = oe(Y), je = oe(Te);
          {
            var Se = (Ve) => {
              var it = _t(), mt = ge(it);
              pn(mt, 17, s, hn, (Lt, Ye, Je) => {
                var Ae = _t(), ze = ge(Ae);
                {
                  var at = (u) => {
                    var d = Qo(), A = ge(d);
                    {
                      var b = (se) => {
                        var $ = Ue(" /");
                        D(se, $);
                      };
                      ee(A, (se) => {
                        Je !== 0 && se(b);
                      });
                    }
                    var I = J(A, 2);
                    {
                      var C = (se) => {
                        var $ = Ue();
                        re(() => ve($, l(Ye).desc)), D(se, $);
                      };
                      ee(I, (se) => {
                        l(Ye).desc != null && se(C);
                      });
                    }
                    var M = J(I, 2);
                    {
                      var N = (se) => {
                        var $ = Ue("-");
                        D(se, $);
                      };
                      ee(M, (se) => {
                        l(Ye).desc != null && l(j).progress_level && l(j).progress_level[Je] != null && se(N);
                      });
                    }
                    var W = J(M, 2);
                    {
                      var ce = (se) => {
                        var $ = Ue();
                        re((Hi) => ve($, `${Hi ?? ""}%`), [
                          () => (100 * (l(j).progress_level[Je] || 0)).toFixed(1)
                        ]), D(se, $);
                      };
                      ee(W, (se) => {
                        l(j).progress_level != null && se(ce);
                      });
                    }
                    D(u, d);
                  };
                  ee(ze, (u) => {
                    (l(Ye).desc != null || l(j).progress_level && l(j).progress_level[Je] != null) && u(at);
                  });
                }
                D(Lt, Ae);
              }), D(Ve, it);
            };
            ee(je, (Ve) => {
              s() != null && Ve(Se);
            });
          }
          var dt = J(Te, 2), de = oe(dt);
          let pt;
          re(() => pt = Ce(de, "", pt, {
            width: `${l(j).last_progress_level * 100}%`,
            transition: l(j).progress_bar_transition
          })), D(V, Y);
        }, br = (V) => {
          {
            let Y = Le(() => c() === "default");
            zo(V, {
              get margin() {
                return l(Y);
              }
            });
          }
        };
        ee(Wt, (V) => {
          l(j).last_progress_level != null ? V(gr) : a() === "full" && V(br, 1);
        });
      }
      var Ot = J(Wt, 2);
      {
        var Mt = (V) => {
          var Y = $o(), Te = ge(Y), je = oe(Te), Se = J(Te, 2);
          Dr(Se, t, "additional-loading-text", {}), re(() => ve(je, f())), D(V, Y);
        };
        ee(Ot, (V) => {
          i() || V(Mt);
        });
      }
      re(() => Oe = tt(ye, 1, "progress-text svelte-124hqw6", null, Oe, {
        "meta-text-center": c() === "center",
        "meta-text": c() === "default"
      })), D(z, K);
    }, It = (z) => {
      var K = tl(), te = ge(K), ie = oe(te);
      {
        let ke = Le(() => t.i18n("common.clear"));
        Cn(ie, {
          get Icon() {
            return Rn;
          },
          get label() {
            return l(ke);
          },
          disabled: !1,
          $$events: {
            click: () => {
              t.on_clear_status?.();
            }
          }
        });
      }
      var ye = J(te, 2), Oe = oe(ye), Me = J(ye, 2);
      Dr(Me, t, "error", {}), re((ke) => ve(Oe, ke), [() => t.i18n("common.error")]), D(z, K);
    };
    ee(qt, (z) => {
      t.status === "pending" ? z(pr) : t.status === "error" && z(It, 1);
    });
  }
  rr(Fe, (z) => T = z, () => T);
  var ht = J(Fe, 2);
  {
    var mr = (z) => {
      var K = rl();
      let te, ie;
      var ye = oe(K), Oe = J(ye);
      {
        var Me = (Ze) => {
          var nt = Ue();
          re(() => ve(nt, `~${l(fe) ?? ""}s
			→ `)), D(Ze, nt);
        };
        ee(Oe, (Ze) => {
          l(we) && Ze(Me);
        });
      }
      var ke = J(Oe);
      re(() => {
        te = tt(K, 1, "cache-indicator svelte-124hqw6", null, te, { "fade-out": l(he) }), ie = Ce(K, "", ie, { position: h() ? "absolute" : "static" }), ve(ye, `⚡ ${l(X) ?? ""}: `), ve(ke, `${l(xe) ?? ""}s`);
      }), D(z, K);
    };
    ee(ht, (z) => {
      l(G) && z(mr);
    });
  }
  re(() => {
    ct = tt(Fe, 1, `wrap ${c() ?? ""} ${a() ?? ""}`, "svelte-124hqw6", ct, {
      "no-click": y() && _(),
      hide: l(Ie),
      translucent: c() === "center" && (t.status === "pending" || t.status === "error") || v() || a() === "minimal" || y(),
      generating: t.status === "generating" && a() === "full",
      border: m()
    }), me = Ce(Fe, "", me, {
      position: h() ? "absolute" : "static",
      padding: h() ? "0" : "var(--size-8) 0"
    });
  }), D(e, De), or();
}
const al = (e) => {
  const t = {};
  for (let r = 0, n = e.length; r < n; r++) {
    const i = e[r];
    for (const a in i)
      t[a] ? t[a] = t[a].concat(i[a]) : t[a] = i[a];
  }
  return t;
}, sl = [
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
], ol = [
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
], ll = [
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
al([
  Object.fromEntries(sl.map((e) => [e, ["*"]])),
  Object.fromEntries(ol.map((e) => [e, ["svg:*"]])),
  Object.fromEntries(ll.map((e) => [e, ["math:*"]]))
]);
jt(["touchstart", "touchmove", "touchend", "click", "keydown"]);
var ul = /* @__PURE__ */ new Set(["$$slots", "$$events", "$$legacy"]), fl = /* @__PURE__ */ pe('<!> <div class="stitch-preview svelte-r41nsf"><div class="canvas-wrap svelte-r41nsf"><canvas tabindex="0" role="application" aria-label="tile stitch preview canvas"></canvas></div> <div class="shortcut-bar svelte-r41nsf">↑↓←→ 步长 · Shift 10x · 拖精细 · Shift+拖粗调 · Space平移 · Esc取消 · Ctrl+Z撤销</div> <div class="status svelte-r41nsf"> </div></div>', 1);
function hl(e, t) {
  lr(t, !0);
  const r = /* @__PURE__ */ Ka(t, ul), n = 4, i = 70, a = 3, o = 30, s = 1e-3, c = 0.25, f = new Ro(r);
  let h, v, m = q(Ut({ tiles: [], selected: 0 })), y = q(Ut([])), _ = q("点击画布以启用键盘"), S = q(!1), B = q("crosshair"), p = q(0), g = q(0), T = q(1), E = q(!1), x = q(!1), H = q(!1), P = q(-1), L = 0, U = 0, G = 0, he = 0, xe = 0, fe = 0, X = 0, we = 0, _e = !1, qe = !1, Ie = -1, Re = 0, rt = 0, ne = !1, Be = "", le = 0, Ee = null, j = [], Z = -1;
  function We(u) {
    return typeof u == "number" ? String(u) + "px" : u || "520px";
  }
  function ae(u, d) {
    const A = Number(u);
    return Number.isFinite(A) ? A : d;
  }
  function De(u, d, A) {
    return Math.max(d, Math.min(A, u));
  }
  function Fe(u) {
    return JSON.parse(JSON.stringify(u || { tiles: [], selected: 0 }));
  }
  function ct(u) {
    var d;
    return {
      index: Math.trunc(ae(u.index, 0)),
      image: (d = u.image) !== null && d !== void 0 ? d : null,
      x: ae(u.x, 0),
      y: ae(u.y, 0),
      width: Math.max(1, ae(u.width, 1)),
      height: Math.max(1, ae(u.height, 1))
    };
  }
  function me() {
    return Math.trunc(ae(l(m).selected, 0));
  }
  function Ge() {
    const u = ae(l(m).nudge_step, 1);
    return u > 0 ? u : 1;
  }
  function Xt() {
    const u = ae(l(m).drag_gain, c);
    return u > 0 ? u : c;
  }
  function qt() {
    return !!l(m).diff_mode;
  }
  function pr() {
    return l(m).show_loupe !== !1;
  }
  function It() {
    var u, d;
    const A = me();
    for (const b of l(y))
      if (b.tile.index === A) return b.tile;
    return (d = (u = l(y)[0]) === null || u === void 0 ? void 0 : u.tile) !== null && d !== void 0 ? d : null;
  }
  function ht() {
    const u = It();
    if (!u) return { dx: 0, dy: 0 };
    const d = l(y).find((A) => A.tile.index === u.index);
    return d ? {
      dx: Math.round(u.x - d.baseX),
      dy: Math.round(u.y - d.baseY)
    } : { dx: 0, dy: 0 };
  }
  function mr() {
    return l(H) && qe ? 1 : Xt();
  }
  function z() {
    const u = h?.getBoundingClientRect();
    return {
      width: Math.max(1, u?.width || 1),
      height: Math.max(1, u?.height || 1)
    };
  }
  function K(u, d) {
    return {
      x: (u - l(p)) / l(T),
      y: (d - l(g)) / l(T)
    };
  }
  function te(u, d) {
    return {
      x: u * l(T) + l(p),
      y: d * l(T) + l(g)
    };
  }
  function ie(u) {
    const d = h.getBoundingClientRect();
    return { x: u.clientX - d.left, y: u.clientY - d.top };
  }
  function ye(u, d) {
    return d ? 90 / 255 : u === 0 ? 1 : 140 / 255;
  }
  function Oe(u, d, A) {
    if (!u) {
      d === le && A(null);
      return;
    }
    const b = new Image();
    b.onload = () => {
      d === le && A(b);
    }, b.onerror = () => {
      d === le && A(null);
    }, b.src = u;
  }
  function Me() {
    Ee && (clearTimeout(Ee), Ee = null);
  }
  function ke() {
    return {
      tiles: l(y).map((u) => ({
        index: u.tile.index,
        x: u.tile.x,
        y: u.tile.y
      })),
      selected: me()
    };
  }
  function Ze(u, d) {
    return u.selected === d.selected && u.tiles.length === d.tiles.length && u.tiles.every((A, b) => {
      const I = d.tiles[b];
      return A.index === I.index && Math.abs(A.x - I.x) < 0.01 && Math.abs(A.y - I.y) < 0.01;
    });
  }
  function nt() {
    const u = ke();
    Z >= 0 && Ze(j[Z], u) || (j = j.slice(0, Z + 1), j.push(u), j.length > o && j.shift(), Z = j.length - 1);
  }
  function Bt() {
    const u = ke();
    Z >= 0 && Ze(j[Z], u) || (j = j.slice(0, Z + 1), j.push(u), j.length > o && j.shift(), Z = j.length - 1);
  }
  function vr(u) {
    for (const d of u.tiles) {
      const A = l(y).find((b) => b.tile.index === d.index);
      A && (A.tile.x = d.x, A.tile.y = d.y);
    }
    w(m, Object.assign(Object.assign({}, l(m)), { selected: u.selected }), !0), w(y, [...l(y)], !0);
  }
  function Wt() {
    if (!l(y).length) {
      w(p, 20), w(g, 20), w(T, 1);
      return;
    }
    let u = 1 / 0, d = 1 / 0, A = -1 / 0, b = -1 / 0;
    for (const se of l(y)) {
      const $ = se.tile;
      u = Math.min(u, $.x), d = Math.min(d, $.y), A = Math.max(A, $.x + $.width), b = Math.max(b, $.y + $.height);
    }
    const I = 40, C = Math.max(1, A - u), M = Math.max(1, b - d), { width: N, height: W } = z(), ce = De(Math.min((N - I * 2) / C, (W - I * 2) / M), 0.05, 8);
    w(T, ce, !0), w(p, (N - (u + A) * ce) / 2), w(g, (W - (d + b) * ce) / 2);
  }
  function gr(u) {
    Me(), le += 1;
    const d = le, A = new Map(l(y).map((M) => [M.tile.index, M])), b = Fe(u), I = Array.isArray(b.tiles) ? b.tiles.map((M) => {
      const N = ct(M);
      if (!N.image) {
        const W = A.get(N.index);
        W?.tile.image && (N.image = W.tile.image);
      }
      return N;
    }) : [];
    w(
      m,
      Object.assign(Object.assign({}, b), {
        tiles: I,
        selected: I.length ? Math.trunc(ae(b.selected, I[0].index)) : 0,
        nudge_step: ae(b.nudge_step, 1),
        diff_mode: !!b.diff_mode,
        show_loupe: b.show_loupe !== !1,
        drag_gain: ae(b.drag_gain, c),
        status: b.status || ""
      }),
      !0
    ), w(H, !1), w(P, -1), w(x, !1), j = [], Z = -1;
    const C = I.map((M) => ({
      tile: Object.assign({}, M),
      image: null,
      ready: !1,
      baseX: M.x,
      baseY: M.y
    }));
    w(y, C, !0), C.length && Bt(), w(_, l(m).status || (I.length ? "点击画布以启用键盘" : "等待 tile 数据"), !0);
    for (let M = 0; M < C.length; M++) {
      const N = C[M];
      Oe(N.tile.image, d, (W) => {
        if (d !== le) return;
        const ce = l(y)[M];
        !ce || ce.tile.index !== N.tile.index || (ce.image = W, ce.ready = !!W, w(y, [...l(y)], !0), de());
      });
    }
    requestAnimationFrame(() => Wt());
  }
  Pe(() => {
    const u = JSON.stringify(f.props.value || null);
    u !== Be && (Be = u, gr(f.props.value));
  }), Ia(() => {
    le += 1, Me();
  });
  function br(u) {
    var d;
    const A = Object.assign(Object.assign({}, l(m)), {
      tiles: l(y).map((b) => Object.assign({}, b.tile)),
      selected: me(),
      status: (d = u ?? l(m).status) !== null && d !== void 0 ? d : ""
    });
    f.props.value = A, Be = JSON.stringify(A);
  }
  function Ot(u, d = !0) {
    w(m, Object.assign(Object.assign({}, l(m)), { status: u }), !0), w(_, u, !0), br(u), d && (Me(), f.dispatch("change")), de();
  }
  function Mt(u, d = 140) {
    Ot(u, !1), Me(), Ee = setTimeout(
      () => {
        Ee = null, f.dispatch("change");
      },
      d
    );
  }
  function V(u, d) {
    for (let A = l(y).length - 1; A >= 0; A--) {
      const b = l(y)[A].tile;
      if (u >= b.x && u <= b.x + b.width && d >= b.y && d <= b.y + b.height)
        return b.index;
    }
    return -1;
  }
  function Y(u, d) {
    u < 0 || (w(m, Object.assign(Object.assign({}, l(m)), { selected: u }), !0), w(_, d || "已选择 tile " + String(u), !0), de());
  }
  function Te(u, d) {
    const A = It();
    if (!A) return;
    nt(), A.x += u, A.y += d, w(y, [...l(y)], !0), Bt();
    const b = ht();
    Mt("微调 tile " + String(A.index) + " → dx=" + String(b.dx) + " dy=" + String(b.dy));
  }
  function je(u, d, A) {
    const b = K(u, d);
    w(T, De(l(T) * A, 0.05, 16), !0);
    const I = te(b.x, b.y);
    w(p, l(p) + (u - I.x)), w(g, l(g) + (d - I.y)), de();
  }
  function Se(u) {
    if (!pr() || !ne) return;
    const { width: d, height: A } = z(), b = De(Re, i + 2, d - i - 2), I = De(rt, i + 2, A - i - 2);
    i / (l(T) * a);
    const C = K(b, I);
    u.save(), u.beginPath(), u.arc(b, I, i, 0, Math.PI * 2), u.clip(), u.fillStyle = "#0f172a", u.fillRect(b - i, I - i, i * 2, i * 2), u.translate(b, I), u.scale(a, a), u.translate(-C.x * l(T) - l(p) / a, -C.y * l(T) - l(g) / a), u.scale(l(T), l(T));
    for (const M of l(y)) {
      if (!M.ready || !M.image) continue;
      const N = M.tile, W = l(H) && l(P) === N.index;
      u.globalAlpha = ye(N.index, W), qt() && N.index !== 0 ? u.globalCompositeOperation = "difference" : u.globalCompositeOperation = "source-over", u.drawImage(M.image, N.x, N.y, N.width, N.height);
    }
    u.restore(), u.save(), u.beginPath(), u.arc(b, I, i, 0, Math.PI * 2), u.strokeStyle = "rgba(255,255,255,0.9)", u.lineWidth = 2, u.stroke(), u.strokeStyle = "rgba(15,23,42,0.85)", u.lineWidth = 1, u.beginPath(), u.moveTo(b - 8, I), u.lineTo(b + 8, I), u.moveTo(b, I - 8), u.lineTo(b, I + 8), u.stroke(), u.restore();
  }
  function dt(u) {
    const d = It(), A = ht(), b = [
      "选中 #" + String(me()),
      "dx " + String(A.dx) + "  dy " + String(A.dy),
      "zoom " + l(T).toFixed(2) + "  gain " + mr().toFixed(2)
    ];
    u.save(), u.font = "600 12px ui-monospace, SFMono-Regular, Menlo, monospace";
    const I = 8, C = 16, M = Math.max(...b.map((W) => u.measureText(W).width)) + I * 2, N = b.length * C + I;
    if (u.fillStyle = "rgba(15,23,42,0.82)", u.fillRect(10, 10, M, N), u.fillStyle = "#e2e8f0", b.forEach((W, ce) => {
      u.fillText(W, 10 + I, 10 + I + (ce + 1) * C - 4);
    }), d) {
      const W = te(d.x, d.y), ce = te(d.x + d.width, d.y + d.height);
      u.strokeStyle = "#22d3ee", u.lineWidth = 2, u.strokeRect(W.x, W.y, ce.x - W.x, ce.y - W.y);
    }
    u.restore();
  }
  function de() {
    if (!h) return;
    const u = window.devicePixelRatio || 1, { width: d, height: A } = z();
    h.width = Math.max(1, Math.round(d * u)), h.height = Math.max(1, Math.round(A * u));
    const b = h.getContext("2d");
    if (b) {
      if (b.setTransform(u, 0, 0, u, 0, 0), b.clearRect(0, 0, d, A), b.fillStyle = "#0f172a", b.fillRect(0, 0, d, A), !l(y).length) {
        b.fillStyle = "#94a3b8", b.font = "16px sans-serif", b.fillText("等待 tile 数据", 24, 40);
        return;
      }
      b.save(), b.translate(l(p), l(g)), b.scale(l(T), l(T));
      for (const I of l(y)) {
        if (!I.ready || !I.image) continue;
        const C = I.tile, M = l(H) && l(P) === C.index;
        b.globalAlpha = ye(C.index, M), qt() && C.index !== 0 ? b.globalCompositeOperation = "difference" : b.globalCompositeOperation = "source-over", b.drawImage(I.image, C.x, C.y, C.width, C.height);
      }
      if (b.restore(), l(H) && l(P) >= 0) {
        const I = l(y).find((C) => C.tile.index === l(P));
        if (I) {
          const C = te(G, he), M = te(G + I.tile.width, he + I.tile.height);
          b.save(), b.setLineDash([6, 4]), b.strokeStyle = "rgba(250,204,21,0.95)", b.lineWidth = 2, b.strokeRect(C.x, C.y, M.x - C.x, M.y - C.y), b.restore();
        }
      }
      dt(b), Se(b);
    }
  }
  function pt() {
    w(S, !0), w(_, "键盘已接管");
  }
  function Ve() {
    w(S, !1), w(E, !1), !l(H) && !l(x) && w(B, "crosshair"), w(_, l(m).status || "点击画布以启用键盘", !0);
  }
  function it() {
    h.focus();
  }
  function mt(u) {
    if (!l(S)) return;
    const d = u.key, A = d.toLowerCase(), b = /* @__PURE__ */ new Set([
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
    if ((b.has(d) || b.has(A) || u.ctrlKey && A === "z") && u.preventDefault(), d === " " || d === "Spacebar") {
      w(E, !0), w(B, "grab");
      return;
    }
    if (d === "Escape") {
      if (l(H)) {
        const M = l(y).find((N) => N.tile.index === l(P));
        M && (M.tile.x = L, M.tile.y = U, w(y, [...l(y)], !0)), w(H, !1), w(P, -1), _e = !1, w(_, "已取消拖动"), de();
      }
      return;
    }
    if (u.ctrlKey && A === "z") {
      Z > 0 && (Z -= 1, vr(j[Z]), Mt("撤销到步骤 " + String(Z + 1)));
      return;
    }
    if (d === "+" || d === "=") {
      const M = z();
      je(M.width / 2, M.height / 2, 1.15);
      return;
    }
    if (d === "-" || d === "_") {
      const M = z();
      je(M.width / 2, M.height / 2, 1 / 1.15);
      return;
    }
    let I = 0, C = 0;
    if ((d === "ArrowLeft" || A === "a") && (I = -1), (d === "ArrowRight" || A === "d") && (I = 1), (d === "ArrowUp" || A === "w") && (C = -1), (d === "ArrowDown" || A === "s") && (C = 1), I !== 0 || C !== 0) {
      const M = Ge() * (u.shiftKey ? 10 : 1);
      Te(I * M, C * M);
    }
  }
  function Lt(u) {
    l(S) && (u.key === " " || u.key === "Spacebar") && (w(E, !1), l(x) || w(B, l(H) ? "grabbing" : "crosshair", !0));
  }
  function Ye(u) {
    if (!h) return;
    const d = ie(u);
    if (xe = d.x, fe = d.y, X = d.x, we = d.y, Re = d.x, rt = d.y, ne = !0, qe = u.shiftKey, u.button === 1 || u.button === 0 && l(E)) {
      w(x, !0), Ie = u.pointerId, w(B, "grabbing"), h.setPointerCapture(u.pointerId);
      return;
    }
    if (u.button !== 0) return;
    const A = K(d.x, d.y), b = V(A.x, A.y);
    if (b >= 0) {
      Y(b);
      const I = l(y).find((C) => C.tile.index === b);
      I && (w(P, b, !0), L = I.tile.x, U = I.tile.y, G = I.tile.x, he = I.tile.y, w(H, !1), _e = !1, Ie = u.pointerId, h.setPointerCapture(u.pointerId));
    } else
      w(P, -1);
    de();
  }
  function Je(u) {
    const d = ie(u);
    if (Re = d.x, rt = d.y, ne = !0, l(x)) {
      w(p, l(p) + (d.x - X)), w(g, l(g) + (d.y - we)), X = d.x, we = d.y, w(B, "grabbing"), de();
      return;
    }
    if (Ie === u.pointerId && l(P) >= 0) {
      const A = Math.hypot(d.x - xe, d.y - fe), b = l(y).find((I) => I.tile.index === l(P));
      if (!b) return;
      if (!l(H) && A >= n && (w(H, !0), nt(), w(B, "grabbing")), l(H)) {
        const I = u.shiftKey ? 1 : Xt(), C = (d.x - X) / l(T) * I, M = (d.y - we) / l(T) * I;
        b.tile.x += C, b.tile.y += M, w(y, [...l(y)], !0), _e = !0;
        const N = ht();
        w(_, "拖动 tile " + String(l(P)) + "  dx=" + String(N.dx) + " dy=" + String(N.dy)), Mt(l(_));
      }
      X = d.x, we = d.y, de();
      return;
    }
    l(E) ? w(B, "grab") : w(B, "crosshair"), de();
  }
  function Ae(u) {
    if (l(x)) {
      w(x, !1);
      try {
        h.releasePointerCapture(u.pointerId);
      } catch {
      }
      w(B, l(E) ? "grab" : "crosshair", !0), Ie = -1, de();
      return;
    }
    if (Ie === u.pointerId && l(P) >= 0) {
      const d = ie(u), A = Math.hypot(d.x - xe, d.y - fe);
      if (!l(H) && A < n)
        Y(l(P), "已选择 tile " + String(l(P))), Ot("已选择 tile " + String(l(P)));
      else if (l(H) && _e) {
        Bt();
        const b = ht();
        Ot("tile " + String(l(P)) + " 对齐 dx=" + String(b.dx) + " dy=" + String(b.dy));
      }
      w(H, !1), w(P, -1), _e = !1, Ie = -1;
      try {
        h.releasePointerCapture(u.pointerId);
      } catch {
      }
      w(B, l(E) ? "grab" : "crosshair", !0), de();
    }
  }
  function ze() {
    ne = !1, !l(x) && !l(H) && w(B, l(E) ? "grab" : "crosshair", !0), de();
  }
  function at(u) {
    u.preventDefault();
    const d = ie(u), A = Math.exp(-u.deltaY * (u.ctrlKey ? s * 4 : s));
    je(d.x, d.y, A);
  }
  {
    let u = Le(() => l(S) ? "focus" : "base");
    Uo(e, {
      get visible() {
        return f.shared.visible;
      },
      variant: "solid",
      get border_mode() {
        return l(u);
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
      children: (d, A) => {
        var b = fl(), I = ge(b);
        il(I, es(
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
        var C = J(I, 2), M = oe(C), N = oe(M);
        let W;
        rr(N, ($) => h = $, () => h), rr(M, ($) => v = $, () => v);
        var ce = J(M, 4), se = oe(ce);
        re(
          ($) => {
            Ce(C, $), Ce(N, "cursor:" + l(B)), W = tt(N, 1, "svelte-r41nsf", null, W, { focused: l(S) }), ve(se, l(_));
          },
          [() => "height:" + We(f.props.height)]
        ), gt("focus", N, pt), gt("blur", N, Ve), Ke("click", N, it), Ke("keydown", N, mt), Ke("keyup", N, Lt), Ke("pointerdown", N, Ye), Ke("pointermove", N, Je), Ke("pointerup", N, Ae), gt("pointercancel", N, Ae), gt("pointerleave", N, ze), gt("wheel", N, at), D(d, b);
      },
      $$slots: { default: !0 }
    });
  }
  or();
}
jt([
  "click",
  "keydown",
  "keyup",
  "pointerdown",
  "pointermove",
  "pointerup"
]);
export {
  hl as default
};

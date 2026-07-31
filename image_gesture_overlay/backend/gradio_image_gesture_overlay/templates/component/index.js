import { i as Zr, g as Fn, o as Hi, n as tt, u as te, s as Ii, r as Nr, m as ot, a as S, b as h, t as Yr, d as Pi, q as Bi, c as Un, e as ut, f as sr, h as tr, j as Ni, T as Oi, k as Li, l as rr, p as lt, v as Jr, w as ft, x as Gn, y as kn, z as jn, A as jt, E as or, B as Gt, C as Vn, D as Te, F as an, G as Mi, H as zn, I as Qr, J as Ci, K as sn, L as Ri, M as Di, N as Ze, O as Xn, P as br, Q as Fi, R as Ui, S as Gi, U as ki, V as We, W as qn, X as on, Y as ln, Z as ji, _ as Vi, $ as zi, a0 as Xi, a1 as qi, a2 as Wi, a3 as Zi, a4 as Yi, a5 as Kr, a6 as Ji, a7 as Wn, a8 as lr, a9 as Qi, aa as Ki, ab as $i, ac as ea, ad as ta, ae as ra, af as $r, ag as na, ah as ia, ai as Ie, aj as Or, ak as Lr, al as aa, am as sa, an as nr, ao as oa, ap as la, aq as ua, ar as fa, as as ca, at as Zn, au as Ct, av as ha, aw as un, ax as da, ay as ve, az as ur, aA as fr, aB as W, aC as st, aD as Y, aE as pa, aF as le, aG as ye, aH as Le, aI as V, aJ as va, aK as ma } from "./render-HrIXAxOW.js";
function ga(e) {
  throw new Error("https://svelte.dev/e/lifecycle_outside_component");
}
const ba = [];
function _a(e, t = !1, r = !1) {
  return Kt(e, /* @__PURE__ */ new Map(), "", ba, null, r);
}
function Kt(e, t, r, n, i = null, a = !1) {
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
    if (Zr(e)) {
      var s = (
        /** @type {Snapshot<any>} */
        Array(e.length)
      );
      t.set(e, s), i !== null && t.set(i, s);
      for (var u = 0; u < e.length; u += 1) {
        var l = e[u];
        u in e && (s[u] = Kt(l, t, r, n, null, a));
      }
      return s;
    }
    if (Fn(e) === Hi) {
      s = {}, t.set(e, s), i !== null && t.set(i, s);
      for (var c of Object.keys(e))
        s[c] = Kt(
          // @ts-expect-error
          e[c],
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
      return Kt(
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
function en(e, t, r) {
  if (e == null)
    return t(void 0), r && r(void 0), tt;
  const n = te(
    () => e.subscribe(
      t,
      // @ts-expect-error
      r
    )
  );
  return n.unsubscribe ? () => n.unsubscribe() : n;
}
const bt = [];
function ya(e, t) {
  return {
    subscribe: Vt(e, t).subscribe
  };
}
function Vt(e, t = tt) {
  let r = null;
  const n = /* @__PURE__ */ new Set();
  function i(s) {
    if (Ii(e, s) && (e = s, r)) {
      const u = !bt.length;
      for (const l of n)
        l[1](), bt.push(l, e);
      if (u) {
        for (let l = 0; l < bt.length; l += 2)
          bt[l][0](bt[l + 1]);
        bt.length = 0;
      }
    }
  }
  function a(s) {
    i(s(
      /** @type {T} */
      e
    ));
  }
  function o(s, u = tt) {
    const l = [s, u];
    return n.add(l), n.size === 1 && (r = t(i, a) || tt), s(
      /** @type {T} */
      e
    ), () => {
      n.delete(l), n.size === 0 && r && (r(), r = null);
    };
  }
  return { set: i, update: a, subscribe: o };
}
function At(e, t, r) {
  const n = !Array.isArray(e), i = n ? [e] : e;
  if (!i.every(Boolean))
    throw new Error("derived() expects stores as input, got a falsy value");
  const a = t.length < 2;
  return ya(r, (o, s) => {
    let u = !1;
    const l = [];
    let c = 0, p = tt;
    const g = () => {
      if (c)
        return;
      p();
      const v = t(n ? l[0] : l, o, s);
      a ? o(v) : p = typeof v == "function" ? v : tt;
    }, w = i.map(
      (v, x) => en(
        v,
        (I) => {
          l[x] = I, c &= ~(1 << x), u && g();
        },
        () => {
          c |= 1 << x;
        }
      )
    );
    return u = !0, g(), function() {
      Nr(w), p(), u = !1;
    };
  });
}
function xa(e) {
  let t;
  return en(e, (r) => t = r)(), t;
}
let Yt = !1, Mr = /* @__PURE__ */ Symbol("unmounted");
function fn(e, t, r) {
  const n = r[t] ??= {
    store: null,
    source: ot(void 0),
    unsubscribe: tt
  };
  if (n.store !== e && !(Mr in r))
    if (n.unsubscribe(), n.store = e ?? null, e == null)
      n.source.v = void 0, n.unsubscribe = tt;
    else {
      var i = !0;
      n.unsubscribe = en(e, (a) => {
        i ? n.source.v = a : S(n.source, a);
      }), i = !1;
    }
  return e && Mr in r ? xa(e) : h(n.source);
}
function Ea() {
  const e = {};
  function t() {
    Yr(() => {
      for (var r in e)
        e[r].unsubscribe();
      Pi(e, Mr, {
        enumerable: !1,
        value: !0
      });
    });
  }
  return [e, t];
}
function wa(e) {
  var t = Yt;
  try {
    return Yt = !1, [e(), Yt];
  } finally {
    Yt = t;
  }
}
function Ta(e, t) {
  if (t) {
    const r = document.body;
    e.autofocus = !0, Bi(() => {
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
function Yn(e) {
  var t = Un("template");
  return t.innerHTML = Aa(e.replaceAll("<!>", "<!---->")), t.content;
}
function Et(e, t) {
  var r = (
    /** @type {Effect} */
    sr
  );
  r.nodes === null && (r.nodes = { start: e, end: t, a: null, t: null });
}
// @__NO_SIDE_EFFECTS__
function ue(e, t) {
  var r = (t & Oi) !== 0, n = (t & Li) !== 0, i, a = !e.startsWith("<!>");
  return () => {
    i === void 0 && (i = Yn(a ? e : "<!>" + e), r || (i = /** @type {TemplateNode} */
    tr(i)));
    var o = (
      /** @type {TemplateNode} */
      n || Ni ? document.importNode(i, !0) : i.cloneNode(!0)
    );
    if (r) {
      var s = (
        /** @type {TemplateNode} */
        tr(o)
      ), u = (
        /** @type {TemplateNode} */
        o.lastChild
      );
      Et(s, u);
    } else
      Et(o, o);
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
        Yn(i)
      ), s = (
        /** @type {Element} */
        tr(o)
      );
      a = /** @type {Element} */
      tr(s);
    }
    var u = (
      /** @type {TemplateNode} */
      a.cloneNode(!0)
    );
    return Et(u, u), u;
  };
}
// @__NO_SIDE_EFFECTS__
function Jn(e, t) {
  return /* @__PURE__ */ Ha(e, t, "svg");
}
function Ge(e = "") {
  {
    var t = ut(e + "");
    return Et(t, t), t;
  }
}
function yt() {
  var e = document.createDocumentFragment(), t = document.createComment(""), r = ut();
  return e.append(t, r), Et(t, r), e;
}
function C(e, t) {
  e !== null && e.before(
    /** @type {Node} */
    t
  );
}
class cr {
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
        rr(n), this.#n.delete(r);
      else {
        var i = this.#e.get(r);
        i && (rr(i.effect), this.#r.set(r, i.effect), this.#e.delete(r), i.fragment.lastChild.remove(), this.anchor.before(i.fragment), n = i.effect);
      }
      for (const [a, o] of this.#t) {
        if (this.#t.delete(a), a === t)
          break;
        const s = this.#e.get(o);
        s && (lt(s.effect), this.#e.delete(o));
      }
      for (const [a, o] of this.#r) {
        if (a === r || this.#n.has(a)) continue;
        const s = () => {
          if (Array.from(this.#t.values()).includes(a)) {
            var l = document.createDocumentFragment();
            kn(o, l), l.append(ut()), this.#e.set(a, { effect: o, fragment: l });
          } else
            lt(o);
          this.#n.delete(a), this.#r.delete(a);
        };
        this.#i || !n ? (this.#n.add(a), Jr(o, s, !1)) : s();
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
      r.includes(n) || (lt(i.effect), this.#e.delete(n));
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
    ), i = jn();
    if (r && !this.#r.has(t) && !this.#e.has(t))
      if (i) {
        var a = document.createDocumentFragment(), o = ut();
        a.append(o), this.#e.set(t, {
          effect: ft(() => r(o)),
          fragment: a
        });
      } else
        this.#r.set(
          t,
          ft(() => r(this.anchor))
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
  var n = new cr(e);
  jt(() => {
    const i = t() ?? null;
    n.ensure(i, i && ((a) => i(a, ...r)));
  }, or);
}
function Pa(e) {
  Gt === null && ga(), Vn && Gt.l !== null ? Ba(Gt).m.push(e) : Te(() => {
    const t = te(e);
    if (typeof t == "function") return (
      /** @type {() => void} */
      t
    );
  });
}
function Ba(e) {
  var t = (
    /** @type {ComponentContextLegacy} */
    e.l
  );
  return t.u ??= { a: [], b: [], m: [] };
}
function Z(e, t, r = !1) {
  var n = new cr(e), i = r ? or : 0;
  function a(o, s) {
    n.ensure(o, s);
  }
  jt(() => {
    var o = !1;
    t((s, u = 0) => {
      o = !0, a(u, s);
    }), o || a(-1, null);
  }, i);
}
function cn(e, t) {
  return t;
}
function Na(e, t, r) {
  for (var n = [], i = t.length, a, o = t.length, s = 0; s < i; s++) {
    let p = t[s];
    Jr(
      p,
      () => {
        if (a) {
          if (a.pending.delete(p), a.done.add(p), a.pending.size === 0) {
            var g = (
              /** @type {Set<EachOutroGroup>} */
              e.outrogroups
            );
            Cr(e, Qr(a.done)), g.delete(a), g.size === 0 && (e.outrogroups = null);
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
      ), c = (
        /** @type {Element} */
        l.parentNode
      );
      Ui(c), c.append(l), e.items.clear();
    }
    Cr(e, t, !u);
  } else
    a = {
      pending: new Set(t),
      done: /* @__PURE__ */ new Set()
    }, (e.outrogroups ??= /* @__PURE__ */ new Set()).add(a);
}
function Cr(e, t, r = !0) {
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
      a.f |= Ze;
      const o = document.createDocumentFragment();
      kn(a, o);
    } else
      lt(t[i], r);
  }
}
var hn;
function dn(e, t, r, n, i, a = null) {
  var o = e, s = /* @__PURE__ */ new Map(), u = null, l = zn(() => {
    var d = r();
    return (
      /** @type {V[]} */
      Zr(d) ? d : d == null ? [] : Qr(d)
    );
  }), c, p = /* @__PURE__ */ new Map(), g = !0;
  function w(d) {
    (I.effect.f & Xn) === 0 && (I.pending.delete(d), I.fallback = u, Oa(I, c, o, t, n), u !== null && (c.length === 0 ? (u.f & Ze) === 0 ? rr(u) : (u.f ^= Ze, Ft(u, null, o)) : Jr(u, () => {
      u = null;
    })));
  }
  function v(d) {
    I.pending.delete(d);
  }
  var x = jt(() => {
    c = /** @type {V[]} */
    h(l);
    for (var d = c.length, m = /* @__PURE__ */ new Set(), E = (
      /** @type {Batch} */
      Gn
    ), b = jn(), _ = 0; _ < d; _ += 1) {
      var B = c[_], H = n(B, _), O = g ? null : s.get(H);
      O ? (O.v && an(O.v, B), O.i && an(O.i, _), b && E.unskip_effect(O.e)) : (O = La(
        s,
        g ? o : hn ??= ut(),
        B,
        H,
        _,
        i,
        t,
        r
      ), g || (O.e.f |= Ze), s.set(H, O)), m.add(H);
    }
    if (d === 0 && a && !u && (g ? u = ft(() => a(o)) : (u = ft(() => a(hn ??= ut())), u.f |= Ze)), d > m.size && Mi(), !g)
      if (p.set(E, m), b) {
        for (const [M, k] of s)
          m.has(M) || E.skip_effect(k.e);
        E.oncommit(w), E.ondiscard(v);
      } else
        w(E);
    h(l);
  }), I = { effect: x, items: s, pending: p, outrogroups: null, fallback: u };
  g = !1;
}
function Rt(e) {
  for (; e !== null && (e.f & Fi) === 0; )
    e = e.next;
  return e;
}
function Oa(e, t, r, n, i) {
  var a = t.length, o = e.items, s = Rt(e.effect.first), u, l = null, c = [], p = [], g, w, v, x;
  for (x = 0; x < a; x += 1) {
    if (g = t[x], w = i(g, x), v = /** @type {EachItem} */
    o.get(w).e, e.outrogroups !== null)
      for (const O of e.outrogroups)
        O.pending.delete(v), O.done.delete(v);
    if ((v.f & br) !== 0 && rr(v), (v.f & Ze) !== 0)
      if (v.f ^= Ze, v === s)
        Ft(v, null, r);
      else {
        var I = l ? l.next : s;
        v === e.effect.last && (e.effect.last = v.prev), v.prev && (v.prev.next = v.next), v.next && (v.next.prev = v.prev), $e(e, l, v), $e(e, v, I), Ft(v, I, r), l = v, c = [], p = [], s = Rt(l.next);
        continue;
      }
    if (v !== s) {
      if (u !== void 0 && u.has(v)) {
        if (c.length < p.length) {
          var d = p[0], m;
          l = d.prev;
          var E = c[0], b = c[c.length - 1];
          for (m = 0; m < c.length; m += 1)
            Ft(c[m], d, r);
          for (m = 0; m < p.length; m += 1)
            u.delete(p[m]);
          $e(e, E.prev, b.next), $e(e, l, E), $e(e, b, d), s = d, l = b, x -= 1, c = [], p = [];
        } else
          u.delete(v), Ft(v, s, r), $e(e, v.prev, v.next), $e(e, v, l === null ? e.effect.first : l.next), $e(e, l, v), l = v;
        continue;
      }
      for (c = [], p = []; s !== null && s !== v; )
        (u ??= /* @__PURE__ */ new Set()).add(s), p.push(s), s = Rt(s.next);
      if (s === null)
        continue;
    }
    (v.f & Ze) === 0 && c.push(v), l = v, s = Rt(v.next);
  }
  if (e.outrogroups !== null) {
    for (const O of e.outrogroups)
      O.pending.size === 0 && (Cr(e, Qr(O.done)), e.outrogroups?.delete(O));
    e.outrogroups.size === 0 && (e.outrogroups = null);
  }
  if (s !== null || u !== void 0) {
    var _ = [];
    if (u !== void 0)
      for (v of u)
        (v.f & br) === 0 && _.push(v);
    for (; s !== null; )
      (s.f & br) === 0 && s !== e.fallback && _.push(s), s = Rt(s.next);
    var B = _.length;
    if (B > 0) {
      var H = null;
      Na(e, _, H);
    }
  }
}
function La(e, t, r, n, i, a, o, s) {
  var u = (o & Ri) !== 0 ? (o & Di) === 0 ? ot(r, !1, !1) : sn(r) : null, l = (o & Ci) !== 0 ? sn(i) : null;
  return {
    v: u,
    i: l,
    e: ft(() => (a(t, u ?? r, l ?? i, s), () => {
      e.delete(n);
    }))
  };
}
function Ft(e, t, r) {
  if (e.nodes)
    for (var n = e.nodes.start, i = e.nodes.end, a = t && (t.f & Ze) === 0 ? (
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
function $e(e, t, r) {
  t === null ? e.effect.first = r : t.next = r, r === null ? e.effect.last = t : r.prev = t;
}
function Rr(e, t, r, n, i) {
  var a = t.$$slots?.[r], o = !1;
  a === !0 && (a = t[r === "default" ? "children" : r], o = !0), a === void 0 || a(e, o ? () => n : n);
}
function Ma(e, t, r) {
  var n = new cr(e);
  jt(() => {
    var i = t() ?? null;
    n.ensure(i, i && ((a) => r(a, i)));
  }, or);
}
const Ca = () => performance.now(), Me = {
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
function Qn() {
  const e = Me.now();
  Me.tasks.forEach((t) => {
    t.c(e) || (Me.tasks.delete(t), t.f());
  }), Me.tasks.size !== 0 && Me.tick(Qn);
}
function Ra(e) {
  let t;
  return Me.tasks.size === 0 && Me.tick(Qn), {
    promise: new Promise((r) => {
      Me.tasks.add(t = { c: e, f: r });
    }),
    abort() {
      Me.tasks.delete(t);
    }
  };
}
function Da(e, t, r, n, i, a) {
  var o = null, s = (
    /** @type {TemplateNode} */
    e
  ), u = new cr(s, !1);
  jt(() => {
    const l = t() || null;
    var c = l === "svg" ? ki : void 0;
    if (l === null) {
      u.ensure(null, null);
      return;
    }
    return u.ensure(l, (p) => {
      if (l) {
        if (o = Un(l, c), Et(o, o), n) {
          var g = null, w = o.appendChild(ut());
          n(o, w), g?.remove();
        }
        sr.nodes.end = o, p.before(o);
      }
    }), () => {
    };
  }, or), Yr(() => {
  });
}
function Fa(e, t, r) {
  We(() => {
    var n = te(() => t(e, r?.()) || {});
    if (n?.destroy)
      return () => (
        /** @type {Function} */
        n.destroy()
      );
  });
}
function Ua(e, t) {
  var r = void 0, n;
  qn(() => {
    r !== (r = t()) && (n && (lt(n), n = null), r && (n = ft(() => {
      We(() => (
        /** @type {(node: Element) => void} */
        r(e)
      ));
    })));
  });
}
function Kn(e) {
  var t, r, n = "";
  if (typeof e == "string" || typeof e == "number") n += e;
  else if (typeof e == "object") if (Array.isArray(e)) {
    var i = e.length;
    for (t = 0; t < i; t++) e[t] && (r = Kn(e[t])) && (n && (n += " "), n += r);
  } else for (r in e) e[r] && (n && (n += " "), n += r);
  return n;
}
function Ga() {
  for (var e, t, r = 0, n = "", i = arguments.length; r < i; r++) (e = arguments[r]) && (t = Kn(e)) && (n && (n += " "), n += t);
  return n;
}
function ka(e) {
  return typeof e == "object" ? Ga(e) : e ?? "";
}
const pn = Array.from(" \t\n\r\f\u00a0\v\uFEFF");
function ja(e, t, r) {
  var n = e == null ? "" : "" + e;
  if (t && (n = n ? n + " " + t : t), r) {
    for (var i of Object.keys(r))
      if (r[i])
        n = n ? n + " " + i : i;
      else if (n.length)
        for (var a = i.length, o = 0; (o = n.indexOf(i, o)) >= 0; ) {
          var s = o + a;
          (o === 0 || pn.includes(n[o - 1])) && (s === n.length || pn.includes(n[s])) ? n = (o === 0 ? "" : n.substring(0, o)) + n.substring(s + 1) : o = s;
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
function _r(e) {
  return e[0] !== "-" || e[1] !== "-" ? e.toLowerCase() : e;
}
function Va(e, t) {
  if (t) {
    var r = "", n, i;
    if (Array.isArray(t) ? (n = t[0], i = t[1]) : n = t, e) {
      e = String(e).replaceAll(/\s*\/\*.*?\*\/\s*/g, "").trim();
      var a = !1, o = 0, s = !1, u = [];
      n && u.push(...Object.keys(n).map(_r)), i && u.push(...Object.keys(i).map(_r));
      var l = 0, c = -1;
      const x = e.length;
      for (var p = 0; p < x; p++) {
        var g = e[p];
        if (s ? g === "/" && e[p - 1] === "*" && (s = !1) : a ? a === g && (a = !1) : g === "/" && e[p + 1] === "*" ? s = !0 : g === '"' || g === "'" ? a = g : g === "(" ? o++ : g === ")" && o--, !s && a === !1 && o === 0) {
          if (g === ":" && c === -1)
            c = p;
          else if (g === ";" || p === x - 1) {
            if (c !== -1) {
              var w = _r(e.substring(l, c).trim());
              if (!u.includes(w)) {
                g !== ";" && p++;
                var v = e.substring(l, p).trim();
                r += " " + v + ";";
              }
            }
            l = p + 1, c = -1;
          }
        }
      }
    }
    return n && (r += vn(n)), i && (r += vn(i, !0)), r = r.trim(), r === "" ? null : r;
  }
  return e == null ? null : String(e);
}
function Ye(e, t, r, n, i, a) {
  var o = (
    /** @type {any} */
    e[on]
  );
  if (o !== r || o === void 0) {
    var s = ja(r, n, a);
    s == null ? e.removeAttribute("class") : t ? e.className = s : e.setAttribute("class", s), e[on] = r;
  } else if (a && i !== a)
    for (var u in a) {
      var l = !!a[u];
      (i == null || l !== !!i[u]) && e.classList.toggle(u, l);
    }
  return a;
}
function yr(e, t = {}, r, n) {
  for (var i in r) {
    var a = r[i];
    t[i] !== a && (r[i] == null ? e.style.removeProperty(i) : e.style.setProperty(i, a, n));
  }
}
function Ce(e, t, r, n) {
  var i = (
    /** @type {any} */
    e[ln]
  );
  if (i !== t) {
    var a = Va(t, n);
    a == null ? e.removeAttribute("style") : e.style.cssText = a, e[ln] = t;
  } else n && (Array.isArray(n) ? (yr(e, r?.[0], n[0]), yr(e, r?.[1], n[1], "important")) : yr(e, r, n));
  return n;
}
function Dr(e, t, r = !1) {
  if (e.multiple) {
    if (t == null)
      return;
    if (!Zr(t))
      return ji();
    for (var n of e.options)
      n.selected = t.includes(mn(n));
    return;
  }
  for (n of e.options) {
    var i = mn(n);
    if (Vi(i, t)) {
      n.selected = !0;
      return;
    }
  }
  (!r || t !== void 0) && (e.selectedIndex = -1);
}
function za(e) {
  var t = new MutationObserver(() => {
    "__value" in e && Dr(e, e.__value);
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
  }), Yr(() => {
    t.disconnect();
  });
}
function mn(e) {
  return "__value" in e ? e.__value : e.value;
}
const Ut = /* @__PURE__ */ Symbol("class"), _t = /* @__PURE__ */ Symbol("style"), $n = /* @__PURE__ */ Symbol("is custom element"), ei = /* @__PURE__ */ Symbol("is html"), Xa = Kr ? "input" : "INPUT", qa = Kr ? "option" : "OPTION", Wa = Kr ? "select" : "SELECT";
function Za(e, t) {
  t ? e.hasAttribute("selected") || e.setAttribute("selected", "") : e.removeAttribute("selected");
}
function xt(e, t, r, n) {
  var i = ti(e);
  i[t] !== (i[t] = r) && (t === "loading" && (e[zi] = r), r == null ? e.removeAttribute(t) : typeof r != "string" && ri(e).includes(t) ? e[t] = r : e.setAttribute(t, r));
}
function Ya(e, t, r, n, i = !1, a = !1) {
  var o = ti(e), s = o[$n], u = !o[ei], l = t || {}, c = e.nodeName === qa;
  for (var p in t)
    p in r || (r[p] = null);
  r.class ? r.class = ka(r.class) : r.class = null, r[_t] && (r.style ??= null);
  var g = ri(e);
  if (e.nodeName === Xa && "type" in r && ("value" in r || "__value" in r)) {
    var w = r.type;
    (w !== l.type || w === void 0 && e.hasAttribute("type")) && (l.type = w, xt(e, "type", w));
  }
  for (const b in r) {
    let _ = r[b];
    if (c && b === "value" && _ == null) {
      e.value = e.__value = "", l[b] = _;
      continue;
    }
    if (b === "class") {
      var v = e.namespaceURI === "http://www.w3.org/1999/xhtml";
      Ye(e, v, _, n, t?.[Ut], r[Ut]), l[b] = _, l[Ut] = r[Ut];
      continue;
    }
    if (b === "style") {
      Ce(e, _, t?.[_t], r[_t]), l[b] = _, l[_t] = r[_t];
      continue;
    }
    var x = l[b];
    if (!(_ === x && !(_ === void 0 && e.hasAttribute(b)))) {
      l[b] = _;
      var I = b[0] + b[1];
      if (I !== "$$")
        if (I === "on") {
          const B = {}, H = "$$" + b;
          let O = b.slice(2);
          var d = ea(O);
          if (Ji(O) && (O = O.slice(0, -7), B.capture = !0), !d && x) {
            if (_ != null) continue;
            e.removeEventListener(O, l[H], B), l[H] = null;
          }
          if (d)
            Wn(O, e, _), lr([O]);
          else if (_ != null) {
            let M = function(k) {
              l[b].call(this, k);
            };
            l[H] = Qi(O, e, M, B);
          }
        } else if (b === "style")
          xt(e, b, _);
        else if (b === "autofocus")
          Ta(
            /** @type {HTMLElement} */
            e,
            !!_
          );
        else if (!s && (b === "__value" || b === "value" && _ != null))
          e.value = e.__value = _;
        else if (b === "selected" && c)
          Za(
            /** @type {HTMLOptionElement} */
            e,
            _
          );
        else {
          var m = b;
          u || (m = Ki(m));
          var E = m === "defaultValue" || m === "defaultChecked";
          if (_ == null && !s && !E)
            if (o[b] = null, m === "value" || m === "checked") {
              let B = (
                /** @type {HTMLInputElement} */
                e
              );
              const H = t === void 0;
              if (m === "value") {
                let O = B.defaultValue;
                B.removeAttribute(m), B.defaultValue = O, B.value = B.__value = H ? O : null;
              } else {
                let O = B.defaultChecked;
                B.removeAttribute(m), B.defaultChecked = O, B.checked = H ? O : !1;
              }
            } else
              e.removeAttribute(b);
          else E || g.includes(m) && (s || typeof _ != "string") ? (e[m] = _, m in o && (o[m] = $i)) : typeof _ != "function" && xt(e, m, _);
        }
    }
  }
  return l;
}
function Ja(e, t, r = [], n = [], i = [], a, o = !1, s = !1) {
  Zi(i, r, n, (u) => {
    var l = void 0, c = {}, p = e.nodeName === Wa, g = !1;
    if (qn(() => {
      var v = t(...u.map(h)), x = Ya(
        e,
        l,
        v,
        a,
        o,
        s
      );
      g && p && "value" in v && Dr(
        /** @type {HTMLSelectElement} */
        e,
        v.value
      );
      for (let d of Object.getOwnPropertySymbols(c))
        v[d] || lt(c[d]);
      for (let d of Object.getOwnPropertySymbols(v)) {
        var I = v[d];
        d.description === Yi && (!l || I !== l[d]) && (c[d] && lt(c[d]), c[d] = ft(() => Ua(e, () => I))), x[d] = I;
      }
      l = x;
    }), p) {
      var w = (
        /** @type {HTMLSelectElement} */
        e
      );
      We(() => {
        Dr(
          w,
          /** @type {Record<string | symbol, any>} */
          l.value,
          !0
        ), za(w);
      });
    }
    g = !0;
  });
}
function ti(e) {
  return (
    /** @type {Record<string | symbol, unknown>} **/
    /** @type {any} */
    e[Xi] ??= {
      [$n]: e.nodeName.includes("-"),
      [ei]: e.namespaceURI === qi
    }
  );
}
var gn = /* @__PURE__ */ new Map();
function ri(e) {
  var t = e.getAttribute("is") || e.nodeName, r = gn.get(t);
  if (r) return r;
  gn.set(t, r = []);
  for (var n, i = e, a = Element.prototype; a !== i; ) {
    n = Wi(i);
    for (var o in n)
      n[o].set && // better safe than sorry, we don't want spread attributes to mess with HTML content
      o !== "innerHTML" && o !== "textContent" && o !== "innerText" && r.push(o);
    i = Fn(i);
  }
  return r;
}
function xr(e, t) {
  return e === t || e?.[$r] === t;
}
function tn(e = {}, t, r, n) {
  var i = (
    /** @type {ComponentContext} */
    Gt.r
  ), a = (
    /** @type {Effect} */
    sr
  );
  return We(() => {
    var o, s;
    return ta(() => {
      o = s, s = [], te(() => {
        xr(r(...s), e) || (t(e, ...s), o && xr(r(...o), e) && t(null, ...o));
      });
    }), () => {
      let u = a;
      for (; u !== i && u.parent !== null && u.parent.f & ra; )
        u = u.parent;
      const l = () => {
        s && xr(r(...s), e) && t(null, ...s);
      }, c = u.teardown;
      u.teardown = () => {
        l(), c?.();
      };
    };
  }), e;
}
function Qa(e = !1) {
  const t = (
    /** @type {ComponentContextLegacy} */
    Gt
  ), r = t.l.u;
  if (!r) return;
  let n = () => Ie(t.s);
  if (e) {
    let i = 0, a = (
      /** @type {Record<string, any>} */
      {}
    );
    const o = Or(() => {
      let s = !1;
      const u = t.s;
      for (const l in u)
        u[l] !== a[l] && (a[l] = u[l], s = !0);
      return s && i++, i;
    });
    n = () => h(o);
  }
  r.b.length && na(() => {
    bn(t, n), Nr(r.b);
  }), Te(() => {
    const i = te(() => r.m.map(ia));
    return () => {
      for (const a of i)
        typeof a == "function" && a();
    };
  }), r.a.length && Te(() => {
    bn(t, n), Nr(r.a);
  });
}
function bn(e, t) {
  if (e.l.s)
    for (const r of e.l.s) h(r);
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
  return new Proxy({ props: e, exclude: t }, Ka);
}
const es = {
  get(e, t) {
    let r = e.props.length;
    for (; r--; ) {
      let n = e.props[r];
      if (Ct(n) && (n = n()), typeof n == "object" && n !== null && t in n) return n[t];
    }
  },
  set(e, t, r) {
    let n = e.props.length;
    for (; n--; ) {
      let i = e.props[n];
      Ct(i) && (i = i());
      const a = Lr(i, t);
      if (a && a.set)
        return a.set(r), !0;
    }
    return !1;
  },
  getOwnPropertyDescriptor(e, t) {
    let r = e.props.length;
    for (; r--; ) {
      let n = e.props[r];
      if (Ct(n) && (n = n()), typeof n == "object" && n !== null && t in n) {
        const i = Lr(n, t);
        return i && !i.configurable && (i.configurable = !0), i;
      }
    }
  },
  has(e, t) {
    if (t === $r || t === Zn) return !1;
    for (let r of e.props)
      if (Ct(r) && (r = r()), r != null && t in r) return !0;
    return !1;
  },
  ownKeys(e) {
    const t = [];
    for (let r of e.props)
      if (Ct(r) && (r = r()), !!r) {
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
function P(e, t, r, n) {
  var i = !Vn || (r & la) !== 0, a = (r & oa) !== 0, o = (r & fa) !== 0, s = (
    /** @type {V} */
    n
  ), u = !0, l = (
    /** @type {Derived<V> | undefined} */
    void 0
  ), c = () => o && i ? (l ??= Or(
    /** @type {() => V} */
    n
  ), h(l)) : (u && (u = !1, s = o ? te(
    /** @type {() => V} */
    n
  ) : (
    /** @type {V} */
    n
  )), s);
  let p;
  if (a) {
    var g = $r in e || Zn in e;
    p = Lr(e, t)?.set ?? (g && t in e ? (b) => e[t] = b : void 0);
  }
  var w, v = !1;
  a ? [w, v] = wa(() => (
    /** @type {V} */
    e[t]
  )) : w = /** @type {V} */
  e[t], w === void 0 && n !== void 0 && (w = c(), p && (i && aa(), p(w)));
  var x;
  if (i ? x = () => {
    var b = (
      /** @type {V} */
      e[t]
    );
    return b === void 0 ? c() : (u = !0, b);
  } : x = () => {
    var b = (
      /** @type {V} */
      e[t]
    );
    return b !== void 0 && (s = /** @type {V} */
    void 0), b === void 0 ? s : b;
  }, i && (r & sa) === 0)
    return x;
  if (p) {
    var I = e.$$legacy;
    return (
      /** @type {() => V} */
      (function(b, _) {
        return arguments.length > 0 ? ((!i || !_ || I || v) && p(_ ? x() : b), b) : x();
      })
    );
  }
  var d = !1, m = ((r & ua) !== 0 ? Or : zn)(() => (d = !1, x()));
  a && h(m);
  var E = (
    /** @type {Effect} */
    sr
  );
  return (
    /** @type {() => V} */
    (function(b, _) {
      if (arguments.length > 0) {
        const B = _ ? h(m) : i && a ? nr(b) : b;
        return S(m, B), d = !0, s !== void 0 && (s = B), b;
      }
      return ca && d || (E.f & Xn) !== 0 ? m.v : h(m);
    })
  );
}
ha();
var rs = /* @__PURE__ */ Jn('<svg class="resize-handle svelte-1stq1b1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><line x1="1" y1="9" x2="9" y2="1" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line><line x1="5" y1="9" x2="9" y2="5" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line></svg>'), _n = /* @__PURE__ */ ue("<!> <!>", 1), ns = /* @__PURE__ */ ue('<div class="placeholder svelte-1stq1b1"></div>');
function is(e, t) {
  fr(t, !1);
  let r = P(t, "height", 8, void 0), n = P(t, "min_height", 8, void 0), i = P(t, "max_height", 8, void 0), a = P(t, "width", 8, void 0), o = P(t, "elem_id", 8, ""), s = P(t, "elem_classes", 24, () => []), u = P(t, "variant", 8, "solid"), l = P(t, "border_mode", 8, "base"), c = P(t, "padding", 8, !0), p = P(t, "type", 8, "normal"), g = P(t, "test_id", 8, void 0), w = P(t, "explicit_call", 8, !1), v = P(t, "container", 8, !0), x = P(t, "visible", 8, !0), I = P(t, "allow_overflow", 8, !0), d = P(t, "overflow_behavior", 8, "auto"), m = P(t, "scale", 8, null), E = P(t, "min_width", 8, 0), b = P(t, "flex", 12, !1), _ = P(t, "resizable", 8, !1), B = P(t, "rtl", 8, !1), H = P(t, "fullscreen", 12, !1), O = P(t, "label", 8, void 0), M = ot(H()), k = ot(), re = p() === "fieldset" ? "fieldset" : "div", xe = ot(0), J = ot(0), N = ot(null);
  function ne(Q) {
    H() && Q.key === "Escape" && H(!1);
  }
  const he = (Q) => {
    if (Q !== void 0) {
      if (typeof Q == "number")
        return Q + "px";
      if (typeof Q == "string")
        return Q;
    }
  }, Re = (Q) => {
    let ge = Q.clientY;
    const Ae = (z) => {
      const ee = z.clientY - ge;
      ge = z.clientY, pa(k, h(k).style.height = `${h(k).offsetHeight + ee}px`);
    }, ke = () => {
      window.removeEventListener("mousemove", Ae), window.removeEventListener("mouseup", ke);
    };
    window.addEventListener("mousemove", Ae), window.addEventListener("mouseup", ke);
  };
  un(
    () => (Ie(H()), h(M), h(k)),
    () => {
      H() !== h(M) && (S(M, H()), H() ? (S(N, h(k).getBoundingClientRect()), S(xe, h(k).offsetHeight), S(J, h(k).offsetWidth), window.addEventListener("keydown", ne)) : (S(N, null), window.removeEventListener("keydown", ne)));
    }
  ), un(() => Ie(x()), () => {
    x() || b(!1);
  }), da(), Qa();
  var De = yt(), Pe = ve(De);
  {
    var Se = (Q) => {
      var ge = _n(), Ae = ve(ge);
      Da(Ae, () => re, !1, (ee, Be) => {
        tn(ee, (fe) => S(k, fe), () => h(k)), Ja(
          ee,
          (fe, ce) => ({
            "data-testid": g(),
            id: o(),
            class: `block ${fe ?? ""}`,
            dir: B() ? "rtl" : "ltr",
            "aria-label": O(),
            style: "",
            [Ut]: {
              hidden: x() === "hidden",
              padded: c(),
              flex: b(),
              border_focus: l() === "focus",
              border_contrast: l() === "contrast",
              "hide-container": !w() && !v(),
              fullscreen: H(),
              animating: H() && h(N) !== null,
              "auto-margin": m() === null
            },
            [_t]: ce
          }),
          [
            () => (Ie(s()), te(() => s()?.join(" ") || "")),
            () => ({
              height: (Ie(H()), Ie(r()), te(() => H() ? void 0 : he(r()))),
              "min-height": (Ie(H()), Ie(n()), te(() => H() ? void 0 : he(n()))),
              "max-height": (Ie(H()), Ie(i()), te(() => H() ? void 0 : he(i()))),
              "--start-top": (h(N), te(() => h(N) ? `${h(N).top}px` : "0px")),
              "--start-left": (h(N), te(() => h(N) ? `${h(N).left}px` : "0px")),
              "--start-width": (h(N), te(() => h(N) ? `${h(N).width}px` : "0px")),
              "--start-height": (h(N), te(() => h(N) ? `${h(N).height}px` : "0px")),
              width: (Ie(H()), Ie(a()), te(() => H() ? void 0 : typeof a() == "number" ? `calc(min(${a()}px, 100%))` : he(a()))),
              "border-style": u(),
              overflow: I() ? d() : "hidden",
              "flex-grow": m(),
              "min-width": `calc(min(${E()}px, 100%))`
            })
          ],
          void 0,
          void 0,
          "svelte-1stq1b1"
        );
        var je = _n(), rt = ve(je);
        Rr(rt, t, "default", {});
        var Ve = W(rt, 2);
        {
          var ht = (fe) => {
            var ce = rs();
            st("mousedown", ce, Re), C(fe, ce);
          };
          Z(Ve, (fe) => {
            _() && fe(ht);
          });
        }
        C(Be, je);
      });
      var ke = W(Ae, 2);
      {
        var z = (ee) => {
          var Be = ns();
          let je;
          Y(() => je = Ce(Be, "", je, {
            height: h(xe) + "px",
            width: h(J) + "px"
          })), C(ee, Be);
        };
        Z(ke, (ee) => {
          H() && ee(z);
        });
      }
      C(Q, ge);
    };
    Z(Pe, (Q) => {
      (x() === !0 || x() === "hidden") && Q(Se);
    });
  }
  C(e, De), ur();
}
var as = /* @__PURE__ */ ue('<span class="svelte-vvirtv"> </span>'), ss = /* @__PURE__ */ ue("<button><!> <div><!> <!></div></button>");
function yn(e, t) {
  let r = P(t, "label", 3, ""), n = P(t, "show_label", 3, !1), i = P(t, "pending", 3, !1), a = P(t, "size", 3, "small"), o = P(t, "padded", 3, !0), s = P(t, "highlight", 3, !1), u = P(t, "disabled", 3, !1), l = P(t, "hasPopup", 3, !1), c = P(t, "color", 3, "var(--block-label-text-color)"), p = P(t, "transparent", 3, !1), g = P(t, "background", 3, "var(--block-background-fill)"), w = P(t, "border", 3, "transparent"), v = Le(() => s() ? "var(--color-accent)" : c());
  var x = ss();
  let I, d;
  var m = le(x);
  {
    var E = (M) => {
      var k = as(), re = le(k);
      Y(() => ye(re, r())), C(M, k);
    };
    Z(m, (M) => {
      n() && M(E);
    });
  }
  var b = W(m, 2);
  let _;
  var B = le(b);
  Ma(B, () => t.Icon, (M, k) => {
    k(M, {});
  });
  var H = W(B, 2);
  {
    var O = (M) => {
      var k = yt(), re = ve(k);
      Ia(re, () => t.children), C(M, k);
    };
    Z(H, (M) => {
      t.children && M(O);
    });
  }
  Y(() => {
    I = Ye(x, 1, "icon-button svelte-vvirtv", null, I, {
      pending: i(),
      padded: o(),
      highlight: s(),
      transparent: p()
    }), x.disabled = u(), xt(x, "aria-label", r()), xt(x, "aria-haspopup", l()), xt(x, "title", r()), d = Ce(x, "", d, {
      "--border-color": w(),
      color: !u() && h(v) ? h(v) : "var(--block-label-text-color)",
      "--bg-color": u() ? "auto" : g()
    }), _ = Ye(b, 1, "svelte-vvirtv", null, _, {
      "x-small": a() === "x-small",
      small: a() === "small",
      large: a() === "large",
      medium: a() === "medium"
    });
  }), Wn("click", x, function(...M) {
    t.onclick?.apply(this, M);
  }), C(e, x);
}
lr(["click"]);
var os = /* @__PURE__ */ Jn('<svg width="100%" height="100%" viewBox="0 0 24 24" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:space="preserve" stroke="currentColor" style="fill-rule:evenodd;clip-rule:evenodd;stroke-linecap:round;stroke-linejoin:round;"><g transform="matrix(1.14096,-0.140958,-0.140958,1.14096,-0.0559523,0.0559523)"><path d="M18,6L6.087,17.913" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></g><path d="M4.364,4.364L19.636,19.636" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></svg>');
function xn(e) {
  var t = os();
  C(e, t);
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
], En = {
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
    primary: En[t][r],
    secondary: En[t][n]
  }
}), {});
function us(e) {
  return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, "default") ? e.default : e;
}
var Er, wn;
function fs() {
  if (wn) return Er;
  wn = 1;
  var e = function(m) {
    return t(m) && !r(m);
  };
  function t(d) {
    return !!d && typeof d == "object";
  }
  function r(d) {
    var m = Object.prototype.toString.call(d);
    return m === "[object RegExp]" || m === "[object Date]" || a(d);
  }
  var n = typeof Symbol == "function" && Symbol.for, i = n ? /* @__PURE__ */ Symbol.for("react.element") : 60103;
  function a(d) {
    return d.$$typeof === i;
  }
  function o(d) {
    return Array.isArray(d) ? [] : {};
  }
  function s(d, m) {
    return m.clone !== !1 && m.isMergeableObject(d) ? x(o(d), d, m) : d;
  }
  function u(d, m, E) {
    return d.concat(m).map(function(b) {
      return s(b, E);
    });
  }
  function l(d, m) {
    if (!m.customMerge)
      return x;
    var E = m.customMerge(d);
    return typeof E == "function" ? E : x;
  }
  function c(d) {
    return Object.getOwnPropertySymbols ? Object.getOwnPropertySymbols(d).filter(function(m) {
      return Object.propertyIsEnumerable.call(d, m);
    }) : [];
  }
  function p(d) {
    return Object.keys(d).concat(c(d));
  }
  function g(d, m) {
    try {
      return m in d;
    } catch {
      return !1;
    }
  }
  function w(d, m) {
    return g(d, m) && !(Object.hasOwnProperty.call(d, m) && Object.propertyIsEnumerable.call(d, m));
  }
  function v(d, m, E) {
    var b = {};
    return E.isMergeableObject(d) && p(d).forEach(function(_) {
      b[_] = s(d[_], E);
    }), p(m).forEach(function(_) {
      w(d, _) || (g(d, _) && E.isMergeableObject(m[_]) ? b[_] = l(_, E)(d[_], m[_], E) : b[_] = s(m[_], E));
    }), b;
  }
  function x(d, m, E) {
    E = E || {}, E.arrayMerge = E.arrayMerge || u, E.isMergeableObject = E.isMergeableObject || e, E.cloneUnlessOtherwiseSpecified = s;
    var b = Array.isArray(m), _ = Array.isArray(d), B = b === _;
    return B ? b ? E.arrayMerge(d, m, E) : v(d, m, E) : s(m, E);
  }
  x.all = function(m, E) {
    if (!Array.isArray(m))
      throw new Error("first argument should be an array");
    return m.reduce(function(b, _) {
      return x(b, _, E);
    }, {});
  };
  var I = x;
  return Er = I, Er;
}
var cs = fs();
const hs = /* @__PURE__ */ us(cs);
var Fr = function(e, t) {
  return Fr = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(r, n) {
    r.__proto__ = n;
  } || function(r, n) {
    for (var i in n) Object.prototype.hasOwnProperty.call(n, i) && (r[i] = n[i]);
  }, Fr(e, t);
};
function hr(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Class extends value " + String(t) + " is not a constructor or null");
  Fr(e, t);
  function r() {
    this.constructor = e;
  }
  e.prototype = t === null ? Object.create(t) : (r.prototype = t.prototype, new r());
}
var U = function() {
  return U = Object.assign || function(t) {
    for (var r, n = 1, i = arguments.length; n < i; n++) {
      r = arguments[n];
      for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (t[a] = r[a]);
    }
    return t;
  }, U.apply(this, arguments);
};
function ds(e, t) {
  var r = {};
  for (var n in e) Object.prototype.hasOwnProperty.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && typeof Object.getOwnPropertySymbols == "function")
    for (var i = 0, n = Object.getOwnPropertySymbols(e); i < n.length; i++)
      t.indexOf(n[i]) < 0 && Object.prototype.propertyIsEnumerable.call(e, n[i]) && (r[n[i]] = e[n[i]]);
  return r;
}
function wr(e, t, r) {
  if (r || arguments.length === 2) for (var n = 0, i = t.length, a; n < i; n++)
    (a || !(n in t)) && (a || (a = Array.prototype.slice.call(t, 0, n)), a[n] = t[n]);
  return e.concat(a || Array.prototype.slice.call(t));
}
function Tr(e, t) {
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
function ni(e, t, r) {
  var n = Array.prototype.slice.call(arguments, 3), i = r(n), a = t.get(i);
  return typeof a > "u" && (a = e.apply(this, n), t.set(i, a)), a;
}
function ii(e, t, r, n, i) {
  return r.bind(t, e, n, i);
}
function ms(e, t) {
  var r = e.length === 1 ? vs : ni;
  return ii(e, this, r, t.cache.create(), t.serializer);
}
function gs(e, t) {
  return ii(e, this, ni, t.cache.create(), t.serializer);
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
}, Sr = {
  variadic: gs
}, R;
(function(e) {
  e[e.EXPECT_ARGUMENT_CLOSING_BRACE = 1] = "EXPECT_ARGUMENT_CLOSING_BRACE", e[e.EMPTY_ARGUMENT = 2] = "EMPTY_ARGUMENT", e[e.MALFORMED_ARGUMENT = 3] = "MALFORMED_ARGUMENT", e[e.EXPECT_ARGUMENT_TYPE = 4] = "EXPECT_ARGUMENT_TYPE", e[e.INVALID_ARGUMENT_TYPE = 5] = "INVALID_ARGUMENT_TYPE", e[e.EXPECT_ARGUMENT_STYLE = 6] = "EXPECT_ARGUMENT_STYLE", e[e.INVALID_NUMBER_SKELETON = 7] = "INVALID_NUMBER_SKELETON", e[e.INVALID_DATE_TIME_SKELETON = 8] = "INVALID_DATE_TIME_SKELETON", e[e.EXPECT_NUMBER_SKELETON = 9] = "EXPECT_NUMBER_SKELETON", e[e.EXPECT_DATE_TIME_SKELETON = 10] = "EXPECT_DATE_TIME_SKELETON", e[e.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE = 11] = "UNCLOSED_QUOTE_IN_ARGUMENT_STYLE", e[e.EXPECT_SELECT_ARGUMENT_OPTIONS = 12] = "EXPECT_SELECT_ARGUMENT_OPTIONS", e[e.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE = 13] = "EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE = 14] = "INVALID_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR = 15] = "EXPECT_SELECT_ARGUMENT_SELECTOR", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR = 16] = "EXPECT_PLURAL_ARGUMENT_SELECTOR", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT = 17] = "EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT = 18] = "EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT", e[e.INVALID_PLURAL_ARGUMENT_SELECTOR = 19] = "INVALID_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_PLURAL_ARGUMENT_SELECTOR = 20] = "DUPLICATE_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_SELECT_ARGUMENT_SELECTOR = 21] = "DUPLICATE_SELECT_ARGUMENT_SELECTOR", e[e.MISSING_OTHER_CLAUSE = 22] = "MISSING_OTHER_CLAUSE", e[e.INVALID_TAG = 23] = "INVALID_TAG", e[e.INVALID_TAG_NAME = 25] = "INVALID_TAG_NAME", e[e.UNMATCHED_CLOSING_TAG = 26] = "UNMATCHED_CLOSING_TAG", e[e.UNCLOSED_TAG = 27] = "UNCLOSED_TAG";
})(R || (R = {}));
var X;
(function(e) {
  e[e.literal = 0] = "literal", e[e.argument = 1] = "argument", e[e.number = 2] = "number", e[e.date = 3] = "date", e[e.time = 4] = "time", e[e.select = 5] = "select", e[e.plural = 6] = "plural", e[e.pound = 7] = "pound", e[e.tag = 8] = "tag";
})(X || (X = {}));
var wt;
(function(e) {
  e[e.number = 0] = "number", e[e.dateTime = 1] = "dateTime";
})(wt || (wt = {}));
function Tn(e) {
  return e.type === X.literal;
}
function xs(e) {
  return e.type === X.argument;
}
function ai(e) {
  return e.type === X.number;
}
function si(e) {
  return e.type === X.date;
}
function oi(e) {
  return e.type === X.time;
}
function li(e) {
  return e.type === X.select;
}
function ui(e) {
  return e.type === X.plural;
}
function Es(e) {
  return e.type === X.pound;
}
function fi(e) {
  return e.type === X.tag;
}
function ci(e) {
  return !!(e && typeof e == "object" && e.type === wt.number);
}
function Ur(e) {
  return !!(e && typeof e == "object" && e.type === wt.dateTime);
}
var hi = /[ \xA0\u1680\u2000-\u200A\u202F\u205F\u3000]/, ws = /(?:[Eec]{1,6}|G{1,5}|[Qq]{1,5}|(?:[yYur]+|U{1,5})|[ML]{1,5}|d{1,2}|D{1,3}|F{1}|[abB]{1,5}|[hkHK]{1,2}|w{1,2}|W{1}|m{1,2}|s{1,2}|[zZOvVxX]{1,4})(?=([^']*'[^']*')*[^']*$)/g;
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
  for (var t = e.split(Ss).filter(function(g) {
    return g.length > 0;
  }), r = [], n = 0, i = t; n < i.length; n++) {
    var a = i[n], o = a.split("/");
    if (o.length === 0)
      throw new Error("Invalid number skeleton");
    for (var s = o[0], u = o.slice(1), l = 0, c = u; l < c.length; l++) {
      var p = c[l];
      if (p.length === 0)
        throw new Error("Invalid number skeleton");
    }
    r.push({ stem: s, options: u });
  }
  return r;
}
function Hs(e) {
  return e.replace(/^(.*?)-/, "");
}
var Sn = /^\.(?:(0+)(\*)?|(#+)|(0+)(#+))$/g, di = /^(@+)?(\+|#+)?[rs]?$/g, Is = /(\*)(0+)|(#+)(0+)|(0+)/g, pi = /^(0+)$/;
function An(e) {
  var t = {};
  return e[e.length - 1] === "r" ? t.roundingPriority = "morePrecision" : e[e.length - 1] === "s" && (t.roundingPriority = "lessPrecision"), e.replace(di, function(r, n, i) {
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
function Ps(e) {
  var t;
  if (e[0] === "E" && e[1] === "E" ? (t = {
    notation: "engineering"
  }, e = e.slice(2)) : e[0] === "E" && (t = {
    notation: "scientific"
  }, e = e.slice(1)), t) {
    var r = e.slice(0, 2);
    if (r === "+!" ? (t.signDisplay = "always", e = e.slice(2)) : r === "+?" && (t.signDisplay = "exceptZero", e = e.slice(2)), !pi.test(e))
      throw new Error("Malformed concise eng/scientific notation");
    t.minimumIntegerDigits = e.length;
  }
  return t;
}
function Hn(e) {
  var t = {}, r = vi(e);
  return r || t;
}
function Bs(e) {
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
        t = U(U(U({}, t), { notation: "scientific" }), i.options.reduce(function(u, l) {
          return U(U({}, u), Hn(l));
        }, {}));
        continue;
      case "engineering":
        t = U(U(U({}, t), { notation: "engineering" }), i.options.reduce(function(u, l) {
          return U(U({}, u), Hn(l));
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
        i.options[0].replace(Is, function(u, l, c, p, g, w) {
          if (l)
            t.minimumIntegerDigits = c.length;
          else {
            if (p && g)
              throw new Error("We currently do not support maximum integer digits");
            if (w)
              throw new Error("We currently do not support exact integer digits");
          }
          return "";
        });
        continue;
    }
    if (pi.test(i.stem)) {
      t.minimumIntegerDigits = i.stem.length;
      continue;
    }
    if (Sn.test(i.stem)) {
      if (i.options.length > 1)
        throw new RangeError("Fraction-precision stems only accept a single optional option");
      i.stem.replace(Sn, function(u, l, c, p, g, w) {
        return c === "*" ? t.minimumFractionDigits = l.length : p && p[0] === "#" ? t.maximumFractionDigits = p.length : g && w ? (t.minimumFractionDigits = g.length, t.maximumFractionDigits = g.length + w.length) : (t.minimumFractionDigits = l.length, t.maximumFractionDigits = l.length), "";
      });
      var a = i.options[0];
      a === "w" ? t = U(U({}, t), { trailingZeroDisplay: "stripIfInteger" }) : a && (t = U(U({}, t), An(a)));
      continue;
    }
    if (di.test(i.stem)) {
      t = U(U({}, t), An(i.stem));
      continue;
    }
    var o = vi(i.stem);
    o && (t = U(U({}, t), o));
    var s = Ps(i.stem);
    s && (t = U(U({}, t), s));
  }
  return t;
}
var Jt = {
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
function Ns(e, t) {
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
  var i = Jt[n || ""] || Jt[r || ""] || Jt["".concat(r, "-001")] || Jt["001"];
  return i[0];
}
var Ar, Ls = new RegExp("^".concat(hi.source, "*")), Ms = new RegExp("".concat(hi.source, "*$"));
function D(e, t) {
  return { start: e, end: t };
}
var Cs = !!String.prototype.startsWith && "_a".startsWith("a", 1), Rs = !!String.fromCodePoint, Ds = !!Object.fromEntries, Fs = !!String.prototype.codePointAt, Us = !!String.prototype.trimStart, Gs = !!String.prototype.trimEnd, ks = !!Number.isSafeInteger, js = ks ? Number.isSafeInteger : function(e) {
  return typeof e == "number" && isFinite(e) && Math.floor(e) === e && Math.abs(e) <= 9007199254740991;
}, Gr = !0;
try {
  var Vs = gi("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  Gr = ((Ar = Vs.exec("a")) === null || Ar === void 0 ? void 0 : Ar[0]) === "a";
} catch {
  Gr = !1;
}
var In = Cs ? (
  // Native
  function(t, r, n) {
    return t.startsWith(r, n);
  }
) : (
  // For IE11
  function(t, r, n) {
    return t.slice(n, n + r.length) === r;
  }
), kr = Rs ? String.fromCodePoint : (
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
), Pn = (
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
), mi = Fs ? (
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
    return t.replace(Ls, "");
  }
), Xs = Gs ? (
  // Native
  function(t) {
    return t.trimEnd();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(Ms, "");
  }
);
function gi(e, t) {
  return new RegExp(e, t);
}
var jr;
if (Gr) {
  var Bn = gi("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  jr = function(t, r) {
    var n;
    Bn.lastIndex = r;
    var i = Bn.exec(t);
    return (n = i[1]) !== null && n !== void 0 ? n : "";
  };
} else
  jr = function(t, r) {
    for (var n = []; ; ) {
      var i = mi(t, r);
      if (i === void 0 || bi(i) || Ys(i))
        break;
      n.push(i), r += i >= 65536 ? 2 : 1;
    }
    return kr.apply(void 0, n);
  };
var qs = (
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
              type: X.pound,
              location: D(s, this.clonePosition())
            });
          } else if (a === 60 && !this.ignoreTag && this.peek() === 47) {
            if (n)
              break;
            return this.error(R.UNMATCHED_CLOSING_TAG, D(this.clonePosition(), this.clonePosition()));
          } else if (a === 60 && !this.ignoreTag && Vr(this.peek() || 0)) {
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
            type: X.literal,
            value: "<".concat(i, "/>"),
            location: D(n, this.clonePosition())
          },
          err: null
        };
      if (this.bumpIf(">")) {
        var a = this.parseMessage(t + 1, r, !0);
        if (a.err)
          return a;
        var o = a.val, s = this.clonePosition();
        if (this.bumpIf("</")) {
          if (this.isEOF() || !Vr(this.char()))
            return this.error(R.INVALID_TAG, D(s, this.clonePosition()));
          var u = this.clonePosition(), l = this.parseTagName();
          return i !== l ? this.error(R.UNMATCHED_CLOSING_TAG, D(u, this.clonePosition())) : (this.bumpSpace(), this.bumpIf(">") ? {
            val: {
              type: X.tag,
              value: i,
              children: o,
              location: D(n, this.clonePosition())
            },
            err: null
          } : this.error(R.INVALID_TAG, D(s, this.clonePosition())));
        } else
          return this.error(R.UNCLOSED_TAG, D(n, this.clonePosition()));
      } else
        return this.error(R.INVALID_TAG, D(n, this.clonePosition()));
    }, e.prototype.parseTagName = function() {
      var t = this.offset();
      for (this.bump(); !this.isEOF() && Zs(this.char()); )
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
      var u = D(n, this.clonePosition());
      return {
        val: { type: X.literal, value: i, location: u },
        err: null
      };
    }, e.prototype.tryParseLeftAngleBracket = function() {
      return !this.isEOF() && this.char() === 60 && (this.ignoreTag || // If at the opening tag or closing tag position, bail.
      !Ws(this.peek() || 0)) ? (this.bump(), "<") : null;
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
      return kr.apply(void 0, r);
    }, e.prototype.tryParseUnquoted = function(t, r) {
      if (this.isEOF())
        return null;
      var n = this.char();
      return n === 60 || n === 123 || n === 35 && (r === "plural" || r === "selectordinal") || n === 125 && t > 0 ? null : (this.bump(), kr(n));
    }, e.prototype.parseArgument = function(t, r) {
      var n = this.clonePosition();
      if (this.bump(), this.bumpSpace(), this.isEOF())
        return this.error(R.EXPECT_ARGUMENT_CLOSING_BRACE, D(n, this.clonePosition()));
      if (this.char() === 125)
        return this.bump(), this.error(R.EMPTY_ARGUMENT, D(n, this.clonePosition()));
      var i = this.parseIdentifierIfPossible().value;
      if (!i)
        return this.error(R.MALFORMED_ARGUMENT, D(n, this.clonePosition()));
      if (this.bumpSpace(), this.isEOF())
        return this.error(R.EXPECT_ARGUMENT_CLOSING_BRACE, D(n, this.clonePosition()));
      switch (this.char()) {
        // Simple argument: `{name}`
        case 125:
          return this.bump(), {
            val: {
              type: X.argument,
              // value does not include the opening and closing braces.
              value: i,
              location: D(n, this.clonePosition())
            },
            err: null
          };
        // Argument with options: `{name, format, ...}`
        case 44:
          return this.bump(), this.bumpSpace(), this.isEOF() ? this.error(R.EXPECT_ARGUMENT_CLOSING_BRACE, D(n, this.clonePosition())) : this.parseArgumentOptions(t, r, i, n);
        default:
          return this.error(R.MALFORMED_ARGUMENT, D(n, this.clonePosition()));
      }
    }, e.prototype.parseIdentifierIfPossible = function() {
      var t = this.clonePosition(), r = this.offset(), n = jr(this.message, r), i = r + n.length;
      this.bumpTo(i);
      var a = this.clonePosition(), o = D(t, a);
      return { value: n, location: o };
    }, e.prototype.parseArgumentOptions = function(t, r, n, i) {
      var a, o = this.clonePosition(), s = this.parseIdentifierIfPossible().value, u = this.clonePosition();
      switch (s) {
        case "":
          return this.error(R.EXPECT_ARGUMENT_TYPE, D(o, u));
        case "number":
        case "date":
        case "time": {
          this.bumpSpace();
          var l = null;
          if (this.bumpIf(",")) {
            this.bumpSpace();
            var c = this.clonePosition(), p = this.parseSimpleArgStyleIfPossible();
            if (p.err)
              return p;
            var g = Xs(p.val);
            if (g.length === 0)
              return this.error(R.EXPECT_ARGUMENT_STYLE, D(this.clonePosition(), this.clonePosition()));
            var w = D(c, this.clonePosition());
            l = { style: g, styleLocation: w };
          }
          var v = this.tryParseArgumentClose(i);
          if (v.err)
            return v;
          var x = D(i, this.clonePosition());
          if (l && In(l?.style, "::", 0)) {
            var I = zs(l.style.slice(2));
            if (s === "number") {
              var p = this.parseNumberSkeletonFromString(I, l.styleLocation);
              return p.err ? p : {
                val: { type: X.number, value: n, location: x, style: p.val },
                err: null
              };
            } else {
              if (I.length === 0)
                return this.error(R.EXPECT_DATE_TIME_SKELETON, x);
              var d = I;
              this.locale && (d = Ns(I, this.locale));
              var g = {
                type: wt.dateTime,
                pattern: d,
                location: l.styleLocation,
                parsedOptions: this.shouldParseSkeletons ? Ts(d) : {}
              }, m = s === "date" ? X.date : X.time;
              return {
                val: { type: m, value: n, location: x, style: g },
                err: null
              };
            }
          }
          return {
            val: {
              type: s === "number" ? X.number : s === "date" ? X.date : X.time,
              value: n,
              location: x,
              style: (a = l?.style) !== null && a !== void 0 ? a : null
            },
            err: null
          };
        }
        case "plural":
        case "selectordinal":
        case "select": {
          var E = this.clonePosition();
          if (this.bumpSpace(), !this.bumpIf(","))
            return this.error(R.EXPECT_SELECT_ARGUMENT_OPTIONS, D(E, U({}, E)));
          this.bumpSpace();
          var b = this.parseIdentifierIfPossible(), _ = 0;
          if (s !== "select" && b.value === "offset") {
            if (!this.bumpIf(":"))
              return this.error(R.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, D(this.clonePosition(), this.clonePosition()));
            this.bumpSpace();
            var p = this.tryParseDecimalInteger(R.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, R.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE);
            if (p.err)
              return p;
            this.bumpSpace(), b = this.parseIdentifierIfPossible(), _ = p.val;
          }
          var B = this.tryParsePluralOrSelectOptions(t, s, r, b);
          if (B.err)
            return B;
          var v = this.tryParseArgumentClose(i);
          if (v.err)
            return v;
          var H = D(i, this.clonePosition());
          return s === "select" ? {
            val: {
              type: X.select,
              value: n,
              options: Pn(B.val),
              location: H
            },
            err: null
          } : {
            val: {
              type: X.plural,
              value: n,
              options: Pn(B.val),
              offset: _,
              pluralType: s === "plural" ? "cardinal" : "ordinal",
              location: H
            },
            err: null
          };
        }
        default:
          return this.error(R.INVALID_ARGUMENT_TYPE, D(o, u));
      }
    }, e.prototype.tryParseArgumentClose = function(t) {
      return this.isEOF() || this.char() !== 125 ? this.error(R.EXPECT_ARGUMENT_CLOSING_BRACE, D(t, this.clonePosition())) : (this.bump(), { val: !0, err: null });
    }, e.prototype.parseSimpleArgStyleIfPossible = function() {
      for (var t = 0, r = this.clonePosition(); !this.isEOF(); ) {
        var n = this.char();
        switch (n) {
          case 39: {
            this.bump();
            var i = this.clonePosition();
            if (!this.bumpUntil("'"))
              return this.error(R.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE, D(i, this.clonePosition()));
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
        return this.error(R.INVALID_NUMBER_SKELETON, r);
      }
      return {
        val: {
          type: wt.number,
          tokens: n,
          location: r,
          parsedOptions: this.shouldParseSkeletons ? Bs(n) : {}
        },
        err: null
      };
    }, e.prototype.tryParsePluralOrSelectOptions = function(t, r, n, i) {
      for (var a, o = !1, s = [], u = /* @__PURE__ */ new Set(), l = i.value, c = i.location; ; ) {
        if (l.length === 0) {
          var p = this.clonePosition();
          if (r !== "select" && this.bumpIf("=")) {
            var g = this.tryParseDecimalInteger(R.EXPECT_PLURAL_ARGUMENT_SELECTOR, R.INVALID_PLURAL_ARGUMENT_SELECTOR);
            if (g.err)
              return g;
            c = D(p, this.clonePosition()), l = this.message.slice(p.offset, this.offset());
          } else
            break;
        }
        if (u.has(l))
          return this.error(r === "select" ? R.DUPLICATE_SELECT_ARGUMENT_SELECTOR : R.DUPLICATE_PLURAL_ARGUMENT_SELECTOR, c);
        l === "other" && (o = !0), this.bumpSpace();
        var w = this.clonePosition();
        if (!this.bumpIf("{"))
          return this.error(r === "select" ? R.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT : R.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT, D(this.clonePosition(), this.clonePosition()));
        var v = this.parseMessage(t + 1, r, n);
        if (v.err)
          return v;
        var x = this.tryParseArgumentClose(w);
        if (x.err)
          return x;
        s.push([
          l,
          {
            value: v.val,
            location: D(w, this.clonePosition())
          }
        ]), u.add(l), this.bumpSpace(), a = this.parseIdentifierIfPossible(), l = a.value, c = a.location;
      }
      return s.length === 0 ? this.error(r === "select" ? R.EXPECT_SELECT_ARGUMENT_SELECTOR : R.EXPECT_PLURAL_ARGUMENT_SELECTOR, D(this.clonePosition(), this.clonePosition())) : this.requiresOtherClause && !o ? this.error(R.MISSING_OTHER_CLAUSE, D(this.clonePosition(), this.clonePosition())) : { val: s, err: null };
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
      var u = D(i, this.clonePosition());
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
      var r = mi(this.message, t);
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
      if (In(this.message, t, this.offset())) {
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
      for (; !this.isEOF() && bi(this.char()); )
        this.bump();
    }, e.prototype.peek = function() {
      if (this.isEOF())
        return null;
      var t = this.char(), r = this.offset(), n = this.message.charCodeAt(r + (t >= 65536 ? 2 : 1));
      return n ?? null;
    }, e;
  })()
);
function Vr(e) {
  return e >= 97 && e <= 122 || e >= 65 && e <= 90;
}
function Ws(e) {
  return Vr(e) || e === 47;
}
function Zs(e) {
  return e === 45 || e === 46 || e >= 48 && e <= 57 || e === 95 || e >= 97 && e <= 122 || e >= 65 && e <= 90 || e == 183 || e >= 192 && e <= 214 || e >= 216 && e <= 246 || e >= 248 && e <= 893 || e >= 895 && e <= 8191 || e >= 8204 && e <= 8205 || e >= 8255 && e <= 8256 || e >= 8304 && e <= 8591 || e >= 11264 && e <= 12271 || e >= 12289 && e <= 55295 || e >= 63744 && e <= 64975 || e >= 65008 && e <= 65533 || e >= 65536 && e <= 983039;
}
function bi(e) {
  return e >= 9 && e <= 13 || e === 32 || e === 133 || e >= 8206 && e <= 8207 || e === 8232 || e === 8233;
}
function Ys(e) {
  return e >= 33 && e <= 35 || e === 36 || e >= 37 && e <= 39 || e === 40 || e === 41 || e === 42 || e === 43 || e === 44 || e === 45 || e >= 46 && e <= 47 || e >= 58 && e <= 59 || e >= 60 && e <= 62 || e >= 63 && e <= 64 || e === 91 || e === 92 || e === 93 || e === 94 || e === 96 || e === 123 || e === 124 || e === 125 || e === 126 || e === 161 || e >= 162 && e <= 165 || e === 166 || e === 167 || e === 169 || e === 171 || e === 172 || e === 174 || e === 176 || e === 177 || e === 182 || e === 187 || e === 191 || e === 215 || e === 247 || e >= 8208 && e <= 8213 || e >= 8214 && e <= 8215 || e === 8216 || e === 8217 || e === 8218 || e >= 8219 && e <= 8220 || e === 8221 || e === 8222 || e === 8223 || e >= 8224 && e <= 8231 || e >= 8240 && e <= 8248 || e === 8249 || e === 8250 || e >= 8251 && e <= 8254 || e >= 8257 && e <= 8259 || e === 8260 || e === 8261 || e === 8262 || e >= 8263 && e <= 8273 || e === 8274 || e === 8275 || e >= 8277 && e <= 8286 || e >= 8592 && e <= 8596 || e >= 8597 && e <= 8601 || e >= 8602 && e <= 8603 || e >= 8604 && e <= 8607 || e === 8608 || e >= 8609 && e <= 8610 || e === 8611 || e >= 8612 && e <= 8613 || e === 8614 || e >= 8615 && e <= 8621 || e === 8622 || e >= 8623 && e <= 8653 || e >= 8654 && e <= 8655 || e >= 8656 && e <= 8657 || e === 8658 || e === 8659 || e === 8660 || e >= 8661 && e <= 8691 || e >= 8692 && e <= 8959 || e >= 8960 && e <= 8967 || e === 8968 || e === 8969 || e === 8970 || e === 8971 || e >= 8972 && e <= 8991 || e >= 8992 && e <= 8993 || e >= 8994 && e <= 9e3 || e === 9001 || e === 9002 || e >= 9003 && e <= 9083 || e === 9084 || e >= 9085 && e <= 9114 || e >= 9115 && e <= 9139 || e >= 9140 && e <= 9179 || e >= 9180 && e <= 9185 || e >= 9186 && e <= 9254 || e >= 9255 && e <= 9279 || e >= 9280 && e <= 9290 || e >= 9291 && e <= 9311 || e >= 9472 && e <= 9654 || e === 9655 || e >= 9656 && e <= 9664 || e === 9665 || e >= 9666 && e <= 9719 || e >= 9720 && e <= 9727 || e >= 9728 && e <= 9838 || e === 9839 || e >= 9840 && e <= 10087 || e === 10088 || e === 10089 || e === 10090 || e === 10091 || e === 10092 || e === 10093 || e === 10094 || e === 10095 || e === 10096 || e === 10097 || e === 10098 || e === 10099 || e === 10100 || e === 10101 || e >= 10132 && e <= 10175 || e >= 10176 && e <= 10180 || e === 10181 || e === 10182 || e >= 10183 && e <= 10213 || e === 10214 || e === 10215 || e === 10216 || e === 10217 || e === 10218 || e === 10219 || e === 10220 || e === 10221 || e === 10222 || e === 10223 || e >= 10224 && e <= 10239 || e >= 10240 && e <= 10495 || e >= 10496 && e <= 10626 || e === 10627 || e === 10628 || e === 10629 || e === 10630 || e === 10631 || e === 10632 || e === 10633 || e === 10634 || e === 10635 || e === 10636 || e === 10637 || e === 10638 || e === 10639 || e === 10640 || e === 10641 || e === 10642 || e === 10643 || e === 10644 || e === 10645 || e === 10646 || e === 10647 || e === 10648 || e >= 10649 && e <= 10711 || e === 10712 || e === 10713 || e === 10714 || e === 10715 || e >= 10716 && e <= 10747 || e === 10748 || e === 10749 || e >= 10750 && e <= 11007 || e >= 11008 && e <= 11055 || e >= 11056 && e <= 11076 || e >= 11077 && e <= 11078 || e >= 11079 && e <= 11084 || e >= 11085 && e <= 11123 || e >= 11124 && e <= 11125 || e >= 11126 && e <= 11157 || e === 11158 || e >= 11159 && e <= 11263 || e >= 11776 && e <= 11777 || e === 11778 || e === 11779 || e === 11780 || e === 11781 || e >= 11782 && e <= 11784 || e === 11785 || e === 11786 || e === 11787 || e === 11788 || e === 11789 || e >= 11790 && e <= 11798 || e === 11799 || e >= 11800 && e <= 11801 || e === 11802 || e === 11803 || e === 11804 || e === 11805 || e >= 11806 && e <= 11807 || e === 11808 || e === 11809 || e === 11810 || e === 11811 || e === 11812 || e === 11813 || e === 11814 || e === 11815 || e === 11816 || e === 11817 || e >= 11818 && e <= 11822 || e === 11823 || e >= 11824 && e <= 11833 || e >= 11834 && e <= 11835 || e >= 11836 && e <= 11839 || e === 11840 || e === 11841 || e === 11842 || e >= 11843 && e <= 11855 || e >= 11856 && e <= 11857 || e === 11858 || e >= 11859 && e <= 11903 || e >= 12289 && e <= 12291 || e === 12296 || e === 12297 || e === 12298 || e === 12299 || e === 12300 || e === 12301 || e === 12302 || e === 12303 || e === 12304 || e === 12305 || e >= 12306 && e <= 12307 || e === 12308 || e === 12309 || e === 12310 || e === 12311 || e === 12312 || e === 12313 || e === 12314 || e === 12315 || e === 12316 || e === 12317 || e >= 12318 && e <= 12319 || e === 12320 || e === 12336 || e === 64830 || e === 64831 || e >= 65093 && e <= 65094;
}
function zr(e) {
  e.forEach(function(t) {
    if (delete t.location, li(t) || ui(t))
      for (var r in t.options)
        delete t.options[r].location, zr(t.options[r].value);
    else ai(t) && ci(t.style) || (si(t) || oi(t)) && Ur(t.style) ? delete t.style.location : fi(t) && zr(t.children);
  });
}
function Js(e, t) {
  t === void 0 && (t = {}), t = U({ shouldParseSkeletons: !0, requiresOtherClause: !0 }, t);
  var r = new qs(e, t).parse();
  if (r.err) {
    var n = SyntaxError(R[r.err.kind]);
    throw n.location = r.err.location, n.originalMessage = r.err.message, n;
  }
  return t?.captureLocation || zr(r.val), r.val;
}
var Tt;
(function(e) {
  e.MISSING_VALUE = "MISSING_VALUE", e.INVALID_VALUE = "INVALID_VALUE", e.MISSING_INTL_API = "MISSING_INTL_API";
})(Tt || (Tt = {}));
var dr = (
  /** @class */
  (function(e) {
    hr(t, e);
    function t(r, n, i) {
      var a = e.call(this, r) || this;
      return a.code = n, a.originalMessage = i, a;
    }
    return t.prototype.toString = function() {
      return "[formatjs Error: ".concat(this.code, "] ").concat(this.message);
    }, t;
  })(Error)
), Nn = (
  /** @class */
  (function(e) {
    hr(t, e);
    function t(r, n, i, a) {
      return e.call(this, 'Invalid values for "'.concat(r, '": "').concat(n, '". Options are "').concat(Object.keys(i).join('", "'), '"'), Tt.INVALID_VALUE, a) || this;
    }
    return t;
  })(dr)
), Qs = (
  /** @class */
  (function(e) {
    hr(t, e);
    function t(r, n, i) {
      return e.call(this, 'Value for "'.concat(r, '" must be of type ').concat(n), Tt.INVALID_VALUE, i) || this;
    }
    return t;
  })(dr)
), Ks = (
  /** @class */
  (function(e) {
    hr(t, e);
    function t(r, n) {
      return e.call(this, 'The intl string context variable "'.concat(r, '" was not provided to the string "').concat(n, '"'), Tt.MISSING_VALUE, n) || this;
    }
    return t;
  })(dr)
), me;
(function(e) {
  e[e.literal = 0] = "literal", e[e.object = 1] = "object";
})(me || (me = {}));
function $s(e) {
  return e.length < 2 ? e : e.reduce(function(t, r) {
    var n = t[t.length - 1];
    return !n || n.type !== me.literal || r.type !== me.literal ? t.push(r) : n.value += r.value, t;
  }, []);
}
function eo(e) {
  return typeof e == "function";
}
function $t(e, t, r, n, i, a, o) {
  if (e.length === 1 && Tn(e[0]))
    return [
      {
        type: me.literal,
        value: e[0].value
      }
    ];
  for (var s = [], u = 0, l = e; u < l.length; u++) {
    var c = l[u];
    if (Tn(c)) {
      s.push({
        type: me.literal,
        value: c.value
      });
      continue;
    }
    if (Es(c)) {
      typeof a == "number" && s.push({
        type: me.literal,
        value: r.getNumberFormat(t).format(a)
      });
      continue;
    }
    var p = c.value;
    if (!(i && p in i))
      throw new Ks(p, o);
    var g = i[p];
    if (xs(c)) {
      (!g || typeof g == "string" || typeof g == "number") && (g = typeof g == "string" || typeof g == "number" ? String(g) : ""), s.push({
        type: typeof g == "string" ? me.literal : me.object,
        value: g
      });
      continue;
    }
    if (si(c)) {
      var w = typeof c.style == "string" ? n.date[c.style] : Ur(c.style) ? c.style.parsedOptions : void 0;
      s.push({
        type: me.literal,
        value: r.getDateTimeFormat(t, w).format(g)
      });
      continue;
    }
    if (oi(c)) {
      var w = typeof c.style == "string" ? n.time[c.style] : Ur(c.style) ? c.style.parsedOptions : n.time.medium;
      s.push({
        type: me.literal,
        value: r.getDateTimeFormat(t, w).format(g)
      });
      continue;
    }
    if (ai(c)) {
      var w = typeof c.style == "string" ? n.number[c.style] : ci(c.style) ? c.style.parsedOptions : void 0;
      w && w.scale && (g = g * (w.scale || 1)), s.push({
        type: me.literal,
        value: r.getNumberFormat(t, w).format(g)
      });
      continue;
    }
    if (fi(c)) {
      var v = c.children, x = c.value, I = i[x];
      if (!eo(I))
        throw new Qs(x, "function", o);
      var d = $t(v, t, r, n, i, a), m = I(d.map(function(_) {
        return _.value;
      }));
      Array.isArray(m) || (m = [m]), s.push.apply(s, m.map(function(_) {
        return {
          type: typeof _ == "string" ? me.literal : me.object,
          value: _
        };
      }));
    }
    if (li(c)) {
      var E = c.options[g] || c.options.other;
      if (!E)
        throw new Nn(c.value, g, Object.keys(c.options), o);
      s.push.apply(s, $t(E.value, t, r, n, i));
      continue;
    }
    if (ui(c)) {
      var E = c.options["=".concat(g)];
      if (!E) {
        if (!Intl.PluralRules)
          throw new dr(`Intl.PluralRules is not available in this environment.
Try polyfilling it using "@formatjs/intl-pluralrules"
`, Tt.MISSING_INTL_API, o);
        var b = r.getPluralRules(t, { type: c.pluralType }).select(g - (c.offset || 0));
        E = c.options[b] || c.options.other;
      }
      if (!E)
        throw new Nn(c.value, g, Object.keys(c.options), o);
      s.push.apply(s, $t(E.value, t, r, n, i, g - (c.offset || 0)));
      continue;
    }
  }
  return $s(s);
}
function to(e, t) {
  return t ? U(U(U({}, e || {}), t || {}), Object.keys(e).reduce(function(r, n) {
    return r[n] = U(U({}, e[n]), t[n] || {}), r;
  }, {})) : e;
}
function ro(e, t) {
  return t ? Object.keys(e).reduce(function(r, n) {
    return r[n] = to(e[n], t[n]), r;
  }, U({}, e)) : e;
}
function Hr(e) {
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
    getNumberFormat: Tr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.NumberFormat).bind.apply(t, wr([void 0], r, !1)))();
    }, {
      cache: Hr(e.number),
      strategy: Sr.variadic
    }),
    getDateTimeFormat: Tr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.DateTimeFormat).bind.apply(t, wr([void 0], r, !1)))();
    }, {
      cache: Hr(e.dateTime),
      strategy: Sr.variadic
    }),
    getPluralRules: Tr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.PluralRules).bind.apply(t, wr([void 0], r, !1)))();
    }, {
      cache: Hr(e.pluralRules),
      strategy: Sr.variadic
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
        var c = l.reduce(function(p, g) {
          return !p.length || g.type !== me.literal || typeof p[p.length - 1] != "string" ? p.push(g.value) : p[p.length - 1] += g.value, p;
        }, []);
        return c.length <= 1 ? c[0] || "" : c;
      }, this.formatToParts = function(u) {
        return $t(a.ast, a.locales, a.formatters, a.formats, u, void 0, a.message);
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
        this.ast = e.__parse(t, U(U({}, s), { locale: this.resolvedLocale }));
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
const et = {}, so = (e, t, r) => r && (t in et || (et[t] = {}), e in et[t] || (et[t][e] = r), r), _i = (e, t) => {
  if (t == null)
    return;
  if (t in et && e in et[t])
    return et[t][e];
  const r = pr(t);
  for (let n = 0; n < r.length; n++) {
    const i = r[n], a = lo(i, e);
    if (a)
      return so(e, t, a);
  }
};
let rn;
const zt = Vt({});
function oo(e) {
  return rn[e] || null;
}
function yi(e) {
  return e in rn;
}
function lo(e, t) {
  if (!yi(e))
    return null;
  const r = oo(e);
  return ao(r, t);
}
function uo(e) {
  if (e == null)
    return;
  const t = pr(e);
  for (let r = 0; r < t.length; r++) {
    const n = t[r];
    if (yi(n))
      return n;
  }
}
function fo(e, ...t) {
  delete et[e], zt.update((r) => (r[e] = hs.all([r[e] || {}, ...t]), r));
}
At(
  [zt],
  ([e]) => Object.keys(e)
);
zt.subscribe((e) => rn = e);
const er = {};
function co(e, t) {
  er[e].delete(t), er[e].size === 0 && delete er[e];
}
function xi(e) {
  return er[e];
}
function ho(e) {
  return pr(e).map((t) => {
    const r = xi(t);
    return [t, r ? [...r] : []];
  }).filter(([, t]) => t.length > 0);
}
function Xr(e) {
  return e == null ? !1 : pr(e).some(
    (t) => {
      var r;
      return (r = xi(t)) == null ? void 0 : r.size;
    }
  );
}
function po(e, t) {
  return Promise.all(
    t.map((n) => (co(e, n), n().then((i) => i.default || i)))
  ).then((n) => fo(e, ...n));
}
const Dt = {};
function Ei(e) {
  if (!Xr(e))
    return e in Dt ? Dt[e] : Promise.resolve();
  const t = ho(e);
  return Dt[e] = Promise.all(
    t.map(
      ([r, n]) => po(r, n)
    )
  ).then(() => {
    if (Xr(e))
      return Ei(e);
    delete Dt[e];
  }), Dt[e];
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
function St() {
  return go;
}
const Ir = Vt(!1);
var bo = Object.defineProperty, _o = Object.defineProperties, yo = Object.getOwnPropertyDescriptors, On = Object.getOwnPropertySymbols, xo = Object.prototype.hasOwnProperty, Eo = Object.prototype.propertyIsEnumerable, Ln = (e, t, r) => t in e ? bo(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, wo = (e, t) => {
  for (var r in t || (t = {}))
    xo.call(t, r) && Ln(e, r, t[r]);
  if (On)
    for (var r of On(t))
      Eo.call(t, r) && Ln(e, r, t[r]);
  return e;
}, To = (e, t) => _o(e, yo(t));
let qr;
const ir = Vt(null);
function Mn(e) {
  return e.split("-").map((t, r, n) => n.slice(0, r + 1).join("-")).reverse();
}
function pr(e, t = St().fallbackLocale) {
  const r = Mn(e);
  return t ? [.../* @__PURE__ */ new Set([...r, ...Mn(t)])] : r;
}
function ct() {
  return qr ?? void 0;
}
ir.subscribe((e) => {
  qr = e ?? void 0, typeof window < "u" && e != null && document.documentElement.setAttribute("lang", e);
});
const So = (e) => {
  if (e && uo(e) && Xr(e)) {
    const { loadingDelay: t } = St();
    let r;
    return typeof window < "u" && ct() != null && t ? r = window.setTimeout(
      () => Ir.set(!0),
      t
    ) : Ir.set(!0), Ei(e).then(() => {
      ir.set(e);
    }).finally(() => {
      clearTimeout(r), Ir.set(!1);
    });
  }
  return ir.set(e);
}, Ht = To(wo({}, ir), {
  set: So
}), vr = (e) => {
  const t = /* @__PURE__ */ Object.create(null);
  return (n) => {
    const i = JSON.stringify(n);
    return i in t ? t[i] : t[i] = e(n);
  };
};
var Ao = Object.defineProperty, ar = Object.getOwnPropertySymbols, wi = Object.prototype.hasOwnProperty, Ti = Object.prototype.propertyIsEnumerable, Cn = (e, t, r) => t in e ? Ao(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, nn = (e, t) => {
  for (var r in t || (t = {}))
    wi.call(t, r) && Cn(e, r, t[r]);
  if (ar)
    for (var r of ar(t))
      Ti.call(t, r) && Cn(e, r, t[r]);
  return e;
}, It = (e, t) => {
  var r = {};
  for (var n in e)
    wi.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && ar)
    for (var n of ar(e))
      t.indexOf(n) < 0 && Ti.call(e, n) && (r[n] = e[n]);
  return r;
};
const kt = (e, t) => {
  const { formats: r } = St();
  if (e in r && t in r[e])
    return r[e][t];
  throw new Error(`[svelte-i18n] Unknown "${t}" ${e} format.`);
}, Ho = vr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = It(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format numbers');
    return n && (i = kt("number", n)), new Intl.NumberFormat(r, i);
  }
), Io = vr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = It(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format dates');
    return n ? i = kt("date", n) : Object.keys(i).length === 0 && (i = kt("date", "short")), new Intl.DateTimeFormat(r, i);
  }
), Po = vr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = It(t, ["locale", "format"]);
    if (r == null)
      throw new Error(
        '[svelte-i18n] A "locale" must be set to format time values'
      );
    return n ? i = kt("time", n) : Object.keys(i).length === 0 && (i = kt("time", "short")), new Intl.DateTimeFormat(r, i);
  }
), Bo = (e = {}) => {
  var t = e, {
    locale: r = ct()
  } = t, n = It(t, [
    "locale"
  ]);
  return Ho(nn({ locale: r }, n));
}, No = (e = {}) => {
  var t = e, {
    locale: r = ct()
  } = t, n = It(t, [
    "locale"
  ]);
  return Io(nn({ locale: r }, n));
}, Oo = (e = {}) => {
  var t = e, {
    locale: r = ct()
  } = t, n = It(t, [
    "locale"
  ]);
  return Po(nn({ locale: r }, n));
}, Lo = vr(
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  (e, t = ct()) => new io(e, t, St().formats, {
    ignoreTag: St().ignoreTag
  })
), Mo = (e, t = {}) => {
  var r, n, i, a;
  let o = t;
  typeof e == "object" && (o = e, e = o.id);
  const {
    values: s,
    locale: u = ct(),
    default: l
  } = o;
  if (u == null)
    throw new Error(
      "[svelte-i18n] Cannot format a message without first setting the initial locale."
    );
  let c = _i(e, u);
  if (!c)
    c = (a = (i = (n = (r = St()).handleMissingMessage) == null ? void 0 : n.call(r, { locale: u, id: e, defaultValue: l })) != null ? i : l) != null ? a : e;
  else if (typeof c != "string")
    return console.warn(
      `[svelte-i18n] Message with id "${e}" must be of type "string", found: "${typeof c}". Gettin its value through the "$format" method is deprecated; use the "json" method instead.`
    ), c;
  if (!s)
    return c;
  let p = c;
  try {
    p = Lo(c, u).format(s);
  } catch (g) {
    g instanceof Error && console.warn(
      `[svelte-i18n] Message "${e}" has syntax error:`,
      g.message
    );
  }
  return p;
}, Co = (e, t) => Oo(t).format(e), Ro = (e, t) => No(t).format(e), Do = (e, t) => Bo(t).format(e), Fo = (e, t = ct()) => _i(e, t);
At([Ht, zt], () => Mo);
At([Ht], () => Co);
At([Ht], () => Ro);
At([Ht], () => Do);
At([Ht, zt], () => Fo);
const Uo = "__i18n__", Go = [
  "label",
  "info",
  "placeholder",
  "description",
  "title",
  "value"
], ko = [
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
  #t = V(nr({}));
  get shared() {
    return h(this.#t);
  }
  set shared(t) {
    S(this.#t, t, !0);
  }
  #r = V(nr({}));
  get props() {
    return h(this.#r);
  }
  set props(t) {
    S(this.#r, t, !0);
  }
  #e = V((t) => t);
  get i18n() {
    return h(this.#e);
  }
  set i18n(t) {
    S(this.#e, t, !0);
  }
  translatable_props = {};
  dispatcher;
  last_update = null;
  shared_props = ko;
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
    for (const n of Go)
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
    ), Te(() => {
      for (const n in t.shared_props)
        this._is_i18n_managed(`shared.${n}`, t.shared_props[n]) || (this.shared[n] = t.shared_props[n]);
      for (const n in t.props)
        this._is_i18n_managed(`props.${n}`, t.props[n]) || (this.props[n] = t.props[n]);
      this.register_component(
        t.shared_props.id,
        // @ts-ignore
        this.set_data.bind(this),
        this.get_data.bind(this)
      ), te(() => {
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
    Te(() => {
      this.mounted || (this.old_value = this.props.value, this.mounted = !0), this.old_value != this.props.value && (this.old_value = this.props.value, this.dispatch("change"));
    });
  }
}
lr(["click"]);
function Pr(e) {
  let t = ["", "k", "M", "G", "T", "P", "E", "Z"], r = 0;
  for (; e > 1e3 && r < t.length - 1; )
    e /= 1e3, r++;
  let n = t[r];
  return (Number.isInteger(e) ? e : e.toFixed(1)) + n;
}
function Rn(e) {
  return Object.prototype.toString.call(e) === "[object Date]";
}
function Wr(e, t, r, n) {
  if (typeof r == "number" || Rn(r)) {
    const i = n - r, a = (r - t) / (e.dt || 1 / 60), o = e.opts.stiffness * i, s = e.opts.damping * a, u = (o - s) * e.inv_mass, l = (a + u) * e.dt;
    return Math.abs(l) < e.opts.precision && Math.abs(i) < e.opts.precision ? n : (e.settled = !1, Rn(r) ? new Date(r.getTime() + l) : r + l);
  } else {
    if (Array.isArray(r))
      return r.map(
        (i, a) => (
          // @ts-ignore
          Wr(e, t[a], r[a], n[a])
        )
      );
    if (typeof r == "object") {
      const i = {};
      for (const a in r)
        i[a] = Wr(e, t[a], r[a], n[a]);
      return i;
    } else
      throw new Error(`Cannot spring ${typeof r} values`);
  }
}
function Dn(e, t = {}) {
  const r = Vt(e), { stiffness: n = 0.15, damping: i = 0.8, precision: a = 0.01 } = t;
  let o, s, u, l = (
    /** @type {T} */
    e
  ), c = (
    /** @type {T | undefined} */
    e
  ), p = 1, g = 0, w = !1;
  function v(I, d = {}) {
    c = I;
    const m = u = {};
    return e == null || d.hard || x.stiffness >= 1 && x.damping >= 1 ? (w = !0, o = Me.now(), l = I, r.set(e = c), Promise.resolve()) : (d.soft && (g = 1 / ((d.soft === !0 ? 0.5 : +d.soft) * 60), p = 0), s || (o = Me.now(), w = !1, s = Ra((E) => {
      if (w)
        return w = !1, s = null, !1;
      p = Math.min(p + g, 1);
      const b = Math.min(E - o, 1e3 / 30), _ = {
        inv_mass: p,
        opts: x,
        settled: !0,
        dt: b * 60 / 1e3
      }, B = Wr(_, l, e, c);
      return o = E, l = /** @type {T} */
      e, r.set(e = /** @type {T} */
      B), _.settled && (s = null), !_.settled;
    })), new Promise((E) => {
      s.promise.then(() => {
        m === u && E();
      });
    }));
  }
  const x = {
    set: v,
    update: (I, d) => v(I(
      /** @type {T} */
      c,
      /** @type {T} */
      e
    ), d),
    subscribe: r.subscribe,
    stiffness: n,
    damping: i,
    precision: a
  };
  return x;
}
var zo = /* @__PURE__ */ ue('<div><svg viewBox="-1200 -1200 3000 3000" fill="none" xmlns="http://www.w3.org/2000/svg" class="svelte-m6d381"><g><path d="M255.926 0.754768L509.702 139.936V221.027L255.926 81.8465V0.754768Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M509.69 139.936L254.981 279.641V361.255L509.69 221.55V139.936Z" fill="#FF7C00" class="svelte-m6d381"></path><path d="M0.250138 139.937L254.981 279.641V361.255L0.250138 221.55V139.937Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M255.923 0.232622L0.236328 139.936V221.55L255.923 81.8469V0.232622Z" fill="#FF7C00" class="svelte-m6d381"></path></g><g><path d="M255.926 141.5L509.702 280.681V361.773L255.926 222.592V141.5Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M509.69 280.679L254.981 420.384V501.998L509.69 362.293V280.679Z" fill="#FF7C00" class="svelte-m6d381"></path><path d="M0.250138 280.681L254.981 420.386V502L0.250138 362.295V280.681Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M255.923 140.977L0.236328 280.68V362.294L255.923 222.591V140.977Z" fill="#FF7C00" class="svelte-m6d381"></path></g></svg></div>');
function Xo(e, t) {
  fr(t, !0);
  const r = () => fn(u, "$top", i), n = () => fn(l, "$bottom", i), [i, a] = Ea();
  var o = this && this.__awaiter || function(E, b, _, B) {
    function H(O) {
      return O instanceof _ ? O : new _(function(M) {
        M(O);
      });
    }
    return new (_ || (_ = Promise))(function(O, M) {
      function k(J) {
        try {
          xe(B.next(J));
        } catch (N) {
          M(N);
        }
      }
      function re(J) {
        try {
          xe(B.throw(J));
        } catch (N) {
          M(N);
        }
      }
      function xe(J) {
        J.done ? O(J.value) : H(J.value).then(k, re);
      }
      xe((B = B.apply(E, b || [])).next());
    });
  };
  let s = P(t, "margin", 3, !0);
  const u = Dn([0, 0]), l = Dn([0, 0]);
  let c = V(!1);
  function p() {
    return o(this, void 0, void 0, function* () {
      yield Promise.all([u.set([125, 140]), l.set([-125, -140])]), yield Promise.all([u.set([-125, 140]), l.set([125, -140])]), yield Promise.all([u.set([-125, 0]), l.set([125, -0])]), yield Promise.all([u.set([125, 0]), l.set([-125, 0])]);
    });
  }
  function g() {
    return o(this, void 0, void 0, function* () {
      yield p(), h(c) || g();
    });
  }
  function w() {
    return o(this, void 0, void 0, function* () {
      yield Promise.all([u.set([125, 0]), l.set([-125, 0])]), g();
    });
  }
  Te(() => (w(), () => {
    S(c, !0);
  }));
  var v = zo();
  let x;
  var I = le(v), d = le(I), m = W(d);
  Y(() => {
    x = Ye(v, 1, "svelte-m6d381", null, x, { margin: s() }), Ce(d, `transform: translate(${r()[0] ?? ""}px, ${r()[1] ?? ""}px);`), Ce(m, `transform: translate(${n()[0] ?? ""}px, ${n()[1] ?? ""}px);`);
  }), C(e, v), ur(), a();
}
var qo = function(e, t, r, n) {
  function i(a) {
    return a instanceof r ? a : new r(function(o) {
      o(a);
    });
  }
  return new (r || (r = Promise))(function(a, o) {
    function s(c) {
      try {
        l(n.next(c));
      } catch (p) {
        o(p);
      }
    }
    function u(c) {
      try {
        l(n.throw(c));
      } catch (p) {
        o(p);
      }
    }
    function l(c) {
      c.done ? a(c.value) : i(c.value).then(s, u);
    }
    l((n = n.apply(e, t || [])).next());
  });
};
let Qt = [], Br = !1;
const Wo = typeof window < "u", Si = Wo ? window.requestAnimationFrame : (e) => {
};
function Zo(e) {
  return qo(this, arguments, void 0, function* (t, r = !0) {
    if (!(window.__gradio_mode__ === "website" || window.__gradio_mode__ !== "app" && r !== !0)) {
      if (Qt.push(t), !Br) Br = !0;
      else return;
      yield va(), Si(() => {
        let n = [0, 0];
        for (let i = 0; i < Qt.length; i++) {
          const o = Qt[i].getBoundingClientRect();
          (i === 0 || o.top + window.scrollY <= n[0]) && (n[0] = o.top + window.scrollY, n[1] = i);
        }
        window.scrollTo({ top: n[0] - 20, behavior: "smooth" }), Br = !1, Qt = [];
      });
    }
  });
}
var Yo = /* @__PURE__ */ ue('<div class="validation-error svelte-124hqw6"> <button class="svelte-124hqw6"><!></button></div>'), Jo = /* @__PURE__ */ ue('<div class="eta-bar svelte-124hqw6"></div>'), Qo = /* @__PURE__ */ ue("<!> ", 1), Ko = /* @__PURE__ */ ue("<!> <!> <!> <!>", 1), $o = /* @__PURE__ */ ue('<div class="progress-level svelte-124hqw6"><div class="progress-level-inner svelte-124hqw6"><!></div> <div class="progress-bar-wrap svelte-124hqw6"><div class="progress-bar svelte-124hqw6"></div></div></div>'), el = /* @__PURE__ */ ue('<p class="loading svelte-124hqw6"> </p> <!>', 1), tl = /* @__PURE__ */ ue("<!> <div><!> <!></div> <!> <!>", 1), rl = /* @__PURE__ */ ue('<div class="clear-status svelte-124hqw6"><!></div> <span class="error svelte-124hqw6"> </span> <!>', 1), nl = /* @__PURE__ */ ue("<div> <!> </div>"), il = /* @__PURE__ */ ue('<div data-testid="status-tracker"><!> <!></div> <!>', 1);
function al(e, t) {
  fr(t, !0);
  let r = P(t, "eta", 3, null), n = P(t, "scroll_to_output", 3, !1), i = P(t, "timer", 3, !0), a = P(t, "show_progress", 3, "full"), o = P(t, "message", 3, null), s = P(t, "progress", 3, null), u = P(t, "variant", 3, "default"), l = P(t, "loading_text", 3, "Loading..."), c = P(t, "absolute", 3, !0), p = P(t, "translucent", 3, !1), g = P(t, "border", 3, !1), w = P(t, "validation_error", 7, null), v = P(t, "show_validation_error", 3, !0), x = P(t, "type", 3, null), I = P(t, "used_cache", 3, null), d = P(t, "cache_duration", 3, null), m = P(t, "avg_time", 3, null), E, b = !1, _ = V(0), B = V(null), H = V(null), O = V(!1), M = V(null), k = V(!1), re = V(!1), xe = V(null), J = V(null), N = V("from cache"), ne = V(!1), he = null, Re = null;
  const De = Le(() => !(v() && w()) && (x() === "input" || !t.status || t.status === "complete" || a() === "hidden" || t.status == "streaming"));
  let Pe = V(0);
  const Se = Le(() => h(H) === null || h(H) <= 0 || !h(Pe) ? 0 : Math.min(h(Pe) / h(H), 1)), Q = Le(() => h(Pe).toFixed(1));
  let ge = Le(() => s() == null), Ae = Le(() => r() !== null && r() !== void 0 ? r() : h(B));
  function ke() {
    Si(() => {
      S(Pe, (performance.now() - h(_)) / 1e3), b && ke();
    });
  }
  let z = Le(() => {
    let j = null;
    s() != null ? j = s().map((ae) => {
      if (ae.index != null && ae.length != null)
        return ae.index / ae.length;
      if (ae.progress != null)
        return ae.progress;
    }) : j = null;
    let K, ie = "";
    return j ? (K = j[j.length - 1], K === 0 ? ie = "0" : ie = "150ms") : K = void 0, {
      progress_level: j,
      last_progress_level: K,
      progress_bar_transition: ie
    };
  });
  function ee() {
    b || (S(B, S(M, null), !0), S(_, performance.now(), !0), b = !0, ke());
  }
  function Be() {
    S(B, S(M, null), !0), b && (b = !1);
  }
  Te(() => {
    t.status === "pending" ? ee() : te(() => {
      Be();
    });
  }), Te(() => {
    E && n() && (t.status === "pending" || t.status === "complete") && Zo(E, t.autoscroll);
  }), Te(() => {
    h(Ae) != null && h(B) !== h(Ae) && (S(H, (performance.now() - h(_)) / 1e3 + h(Ae)), S(M, h(H).toFixed(1), !0), S(B, h(Ae), !0));
  });
  function je() {
    S(O, !1);
  }
  Te(() => {
    te(() => {
      je();
    }), t.status === "error" && o() && S(O, !0);
  }), Te(() => {
    t.status === "complete" && x() === "output" && I() && d() != null && (S(xe, d().toFixed(1), !0), S(N, I() === "full" ? "from cache" : "used cache", !0), S(ne, m() != null && m() > d() && m() > 0, !0), S(J, h(ne) ? m().toFixed(1) : null, !0), S(k, !0), S(re, !1), he && clearTimeout(he), Re && clearTimeout(Re), he = setTimeout(
      () => {
        S(re, !0), Re = setTimeout(
          () => {
            S(k, !1), S(re, !1);
          },
          500
        );
      },
      1750
    ));
  });
  var rt = il(), Ve = ve(rt);
  let ht, fe;
  var ce = le(Ve);
  {
    var Fe = (j) => {
      var K = Yo(), ie = le(K), ae = W(ie), Ee = le(ae);
      {
        let be = Le(() => t.i18n ? t.i18n("common.clear") : "Clear");
        yn(Ee, {
          get Icon() {
            return xn;
          },
          get label() {
            return h(be);
          },
          disabled: !1,
          size: "x-small",
          background: "var(--background-fill-primary)",
          color: "var(--error-background-text)",
          border: "var(--border-color-primary)",
          onclick: () => w(null)
        });
      }
      Y(() => ye(ie, `${w() ?? ""} `)), C(j, K);
    };
    Z(ce, (j) => {
      w() && v() && j(Fe);
    });
  }
  var Pt = W(ce, 2);
  {
    var Xt = (j) => {
      var K = tl(), ie = ve(K);
      {
        var ae = (G) => {
          var q = Jo();
          let He;
          Y(() => He = Ce(q, "", He, {
            transform: `translateX(${(h(Se) || 0) * 100 - 100}%)`
          })), C(G, q);
        };
        Z(ie, (G) => {
          u() === "default" && h(ge) && a() === "full" && G(ae);
        });
      }
      var Ee = W(ie, 2);
      let be;
      var ze = le(Ee);
      {
        var Ue = (G) => {
          var q = yt(), He = ve(q);
          dn(He, 17, s, cn, (nt, _e) => {
            var it = yt(), Mt = ve(it);
            {
              var Je = (Xe) => {
                var at = Qo(), vt = ve(at);
                {
                  var mt = (Ne) => {
                    var f = Ge();
                    Y((y, T) => ye(f, `${y ?? ""}/${T ?? ""}`), [
                      () => Pr(h(_e).index || 0),
                      () => Pr(h(_e).length)
                    ]), C(Ne, f);
                  }, qe = (Ne) => {
                    var f = Ge();
                    Y((y) => ye(f, y), [() => Pr(h(_e).index || 0)]), C(Ne, f);
                  };
                  Z(vt, (Ne) => {
                    h(_e).length != null ? Ne(mt) : Ne(qe, -1);
                  });
                }
                var Qe = W(vt);
                Y(() => ye(Qe, ` ${h(_e).unit ?? ""} |  `)), C(Xe, at);
              };
              Z(Mt, (Xe) => {
                h(_e).index != null && Xe(Je);
              });
            }
            C(nt, it);
          }), C(G, q);
        }, se = (G) => {
          var q = Ge();
          Y(() => ye(q, `queue: ${t.queue_position + 1}/${t.queue_size ?? ""} |`)), C(G, q);
        }, dt = (G) => {
          var q = Ge("processing |");
          C(G, q);
        };
        Z(ze, (G) => {
          s() ? G(Ue) : t.queue_position !== null && t.queue_size !== void 0 && t.queue_position >= 0 ? G(se, 1) : t.queue_position === 0 && G(dt, 2);
        });
      }
      var Bt = W(ze, 2);
      {
        var Nt = (G) => {
          var q = Ge();
          Y(() => ye(q, `${h(Q) ?? ""}${r() ? `/${h(M)}` : ""}s`)), C(G, q);
        };
        Z(Bt, (G) => {
          i() && G(Nt);
        });
      }
      var pt = W(Ee, 2);
      {
        var Ot = (G) => {
          var q = $o(), He = le(q), nt = le(He);
          {
            var _e = (Xe) => {
              var at = yt(), vt = ve(at);
              dn(vt, 17, s, cn, (mt, qe, Qe) => {
                var Ne = yt(), f = ve(Ne);
                {
                  var y = (T) => {
                    var A = Ko(), L = ve(A);
                    {
                      var F = (oe) => {
                        var Oe = Ge(" /");
                        C(oe, Oe);
                      };
                      Z(L, (oe) => {
                        Qe !== 0 && oe(F);
                      });
                    }
                    var $ = W(L, 2);
                    {
                      var we = (oe) => {
                        var Oe = Ge();
                        Y(() => ye(Oe, h(qe).desc)), C(oe, Oe);
                      };
                      Z($, (oe) => {
                        h(qe).desc != null && oe(we);
                      });
                    }
                    var gt = W($, 2);
                    {
                      var de = (oe) => {
                        var Oe = Ge("-");
                        C(oe, Oe);
                      };
                      Z(gt, (oe) => {
                        h(qe).desc != null && h(z).progress_level && h(z).progress_level[Qe] != null && oe(de);
                      });
                    }
                    var pe = W(gt, 2);
                    {
                      var Ke = (oe) => {
                        var Oe = Ge();
                        Y((Ai) => ye(Oe, `${Ai ?? ""}%`), [
                          () => (100 * (h(z).progress_level[Qe] || 0)).toFixed(1)
                        ]), C(oe, Oe);
                      };
                      Z(pe, (oe) => {
                        h(z).progress_level != null && oe(Ke);
                      });
                    }
                    C(T, A);
                  };
                  Z(f, (T) => {
                    (h(qe).desc != null || h(z).progress_level && h(z).progress_level[Qe] != null) && T(y);
                  });
                }
                C(mt, Ne);
              }), C(Xe, at);
            };
            Z(nt, (Xe) => {
              s() != null && Xe(_e);
            });
          }
          var it = W(He, 2), Mt = le(it);
          let Je;
          Y(() => Je = Ce(Mt, "", Je, {
            width: `${h(z).last_progress_level * 100}%`,
            transition: h(z).progress_bar_transition
          })), C(G, q);
        }, Wt = (G) => {
          {
            let q = Le(() => u() === "default");
            Xo(G, {
              get margin() {
                return h(q);
              }
            });
          }
        };
        Z(pt, (G) => {
          h(z).last_progress_level != null ? G(Ot) : a() === "full" && G(Wt, 1);
        });
      }
      var Zt = W(pt, 2);
      {
        var Lt = (G) => {
          var q = el(), He = ve(q), nt = le(He), _e = W(He, 2);
          Rr(_e, t, "additional-loading-text", {}), Y(() => ye(nt, l())), C(G, q);
        };
        Z(Zt, (G) => {
          i() || G(Lt);
        });
      }
      Y(() => be = Ye(Ee, 1, "progress-text svelte-124hqw6", null, be, {
        "meta-text-center": u() === "center",
        "meta-text": u() === "default"
      })), C(j, K);
    }, mr = (j) => {
      var K = rl(), ie = ve(K), ae = le(ie);
      {
        let Ue = Le(() => t.i18n("common.clear"));
        yn(ae, {
          get Icon() {
            return xn;
          },
          get label() {
            return h(Ue);
          },
          disabled: !1,
          $$events: {
            click: () => {
              t.on_clear_status?.();
            }
          }
        });
      }
      var Ee = W(ie, 2), be = le(Ee), ze = W(Ee, 2);
      Rr(ze, t, "error", {}), Y((Ue) => ye(be, Ue), [() => t.i18n("common.error")]), C(j, K);
    };
    Z(Pt, (j) => {
      t.status === "pending" ? j(Xt) : t.status === "error" && j(mr, 1);
    });
  }
  tn(Ve, (j) => E = j, () => E);
  var qt = W(Ve, 2);
  {
    var gr = (j) => {
      var K = nl();
      let ie, ae;
      var Ee = le(K), be = W(Ee);
      {
        var ze = (se) => {
          var dt = Ge();
          Y(() => ye(dt, `~${h(J) ?? ""}s
			→ `)), C(se, dt);
        };
        Z(be, (se) => {
          h(ne) && se(ze);
        });
      }
      var Ue = W(be);
      Y(() => {
        ie = Ye(K, 1, "cache-indicator svelte-124hqw6", null, ie, { "fade-out": h(re) }), ae = Ce(K, "", ae, { position: c() ? "absolute" : "static" }), ye(Ee, `⚡ ${h(N) ?? ""}: `), ye(Ue, `${h(xe) ?? ""}s`);
      }), C(j, K);
    };
    Z(qt, (j) => {
      h(k) && j(gr);
    });
  }
  Y(() => {
    ht = Ye(Ve, 1, `wrap ${u() ?? ""} ${a() ?? ""}`, "svelte-124hqw6", ht, {
      "no-click": w() && v(),
      hide: h(De),
      translucent: u() === "center" && (t.status === "pending" || t.status === "error") || p() || a() === "minimal" || w(),
      generating: t.status === "generating" && a() === "full",
      border: g()
    }), fe = Ce(Ve, "", fe, {
      position: c() ? "absolute" : "static",
      padding: c() ? "0" : "var(--size-8) 0"
    });
  }), C(e, rt), ur();
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
lr(["touchstart", "touchmove", "touchend", "click", "keydown"]);
var fl = /* @__PURE__ */ new Set(["$$slots", "$$events", "$$legacy"]), cl = /* @__PURE__ */ ue("<div></div>"), hl = /* @__PURE__ */ ue('<!> <div role="application" aria-label="Image gesture overlay"><!></div>', 1);
function pl(e, t) {
  fr(t, !0);
  var r = this && this.__awaiter || function(f, y, T, A) {
    function L(F) {
      return F instanceof T ? F : new T(function($) {
        $(F);
      });
    }
    return new (T || (T = Promise))(function(F, $) {
      function we(pe) {
        try {
          de(A.next(pe));
        } catch (Ke) {
          $(Ke);
        }
      }
      function gt(pe) {
        try {
          de(A.throw(pe));
        } catch (Ke) {
          $(Ke);
        }
      }
      function de(pe) {
        pe.done ? F(pe.value) : L(pe.value).then(we, gt);
      }
      de((A = A.apply(f, y || [])).next());
    });
  };
  const n = /* @__PURE__ */ $a(t, fl), i = new Vo(n), a = 4, o = 500;
  let s, u = V(nr({})), l = V(1), c = V(1), p = V(0), g = V(0), w = V(!1), v = null, x = "mouse", I = null, d = V(null), m = V(null), E = V(null), b = V(null), _ = null, B = null, H = 0, O = "", M = "", k = "", re = !1, xe = "", J = null, N = null, ne = null, he = null, Re = null, De = null, Pe = 0, Se = "", Q = "", ge = "";
  function Ae(f) {
    return JSON.parse(JSON.stringify(f || {}));
  }
  function ke(f) {
    return typeof f == "number" ? `${f}px` : f || "320px";
  }
  function z() {
    return String(i.props.target_elem_id || "").replace(/^#/, "").trim();
  }
  function ee() {
    return h(u).server_view || {};
  }
  function Be() {
    const f = String(ee().interaction || "auto").toLowerCase();
    return f === "workspace" || f === "crop" || f === "drag" || f === "bbox" || f === "click" || f === "point" || f === "polygon" || f === "disabled" ? f : "auto";
  }
  function je(f) {
    return f === "touch" ? 12 : f === "pen" ? 7 : 4;
  }
  function rt(f) {
    return f === "touch" ? 12 : f === "pen" ? 7 : 5;
  }
  function Ve() {
    return String(ee().selection_state || "").toLowerCase();
  }
  function ht() {
    return ["auto", "workspace", "click", "point", "polygon"].includes(Be());
  }
  function fe() {
    return ["auto", "workspace", "crop", "drag", "bbox"].includes(Be());
  }
  function ce() {
    const f = Number(ee().natural_width || 0);
    return Number.isFinite(f) && f > 0 ? f : 0;
  }
  function Fe() {
    const f = Number(ee().natural_height || 0);
    return Number.isFinite(f) && f > 0 ? f : 0;
  }
  function Pt() {
    return ee().enabled === !0 && Be() !== "disabled" && (!z() || h(w)) && ce() > 0 && Fe() > 0;
  }
  function Xt() {
    const f = ee();
    return JSON.stringify([
      f.image_id || "",
      f.image_sha256 || "",
      ce(),
      Fe()
    ]);
  }
  function mr() {
    const f = ee();
    return JSON.stringify([
      f.enabled === !0,
      Xt(),
      Number(f.revision || 0),
      Be()
    ]);
  }
  function qt(f) {
    if (!Array.isArray(f) || f.length !== 2) return null;
    const y = Number(f[0]), T = Number(f[1]);
    return Number.isFinite(y) && Number.isFinite(T) ? { x: y, y: T } : null;
  }
  function gr() {
    const f = v;
    v = null, f !== null && s?.hasPointerCapture(f) && s.releasePointerCapture(f), x = "mouse", I = null, S(d, null), S(m, null), _ = null, B = null, H = 0;
  }
  function j() {
    De !== null && clearTimeout(De), De = null;
  }
  function K(f) {
    De = null, !(f !== M || !Se) && (Se = "", N && Bt(N));
  }
  function ie(f) {
    if (j(), S(w, !1), Se = f, Q = "", ge = "", Pe += 1, f) {
      const y = M;
      De = setTimeout(() => K(y), o);
    }
  }
  function ae(f) {
    const y = ge || be(N), T = Ae(f);
    S(u, T, !0);
    const A = Xt(), L = mr(), F = A !== M;
    L !== k && gr(), k = L, F && (M = A, ie(y));
    const we = T.client_intent || {};
    we.gesture === "drag" ? (S(E, qt(we.start_xy), !0), S(b, qt(we.end_xy), !0)) : (S(E, null), S(b, null)), re && z() && (F ? pt() : se());
  }
  Te(() => {
    const f = JSON.stringify(i.props.value || null);
    f !== O && (O = f, ae(i.props.value));
  }), Te(() => {
    const f = z();
    f !== xe && (xe = f, re && Ot());
  });
  function Ee(f) {
    const y = () => {
      if (z()) return;
      const A = f.getBoundingClientRect();
      S(l, Math.max(1, A.width), !0), S(c, Math.max(1, A.height), !0);
    }, T = new ResizeObserver(y);
    return T.observe(f), y(), { destroy: () => T.disconnect() };
  }
  function be(f) {
    return String(f?.currentSrc || f?.src || "");
  }
  function ze() {
    j(), Re !== null && clearTimeout(Re), Re = null, ne?.disconnect(), ne = null, he?.disconnect(), he = null, N?.removeEventListener("load", Nt), J = null, N = null, S(w, !1), Pe += 1, Se = "", Q = "", ge = "";
  }
  function Ue(f, y) {
    if (!y || y === "center") return f / 2;
    if (y === "left" || y === "top") return 0;
    if (y === "right" || y === "bottom") return f;
    if (y.endsWith("%")) {
      const A = Number.parseFloat(y) / 100;
      return Number.isFinite(A) ? f * A : f / 2;
    }
    const T = Number.parseFloat(y);
    return Number.isFinite(T) ? T : f / 2;
  }
  function se() {
    const f = be(N);
    if (!N || !z() || Q !== M || !f || f !== ge) {
      S(w, !1);
      return;
    }
    const y = N.getBoundingClientRect(), T = ce(), A = Fe();
    if (y.width <= 0 || y.height <= 0 || T <= 0 || A <= 0) {
      S(w, !1);
      return;
    }
    const L = Math.min(y.width / T, y.height / A), F = T * L, $ = A * L, we = getComputedStyle(N).objectPosition.split(/\s+/);
    S(p, y.left + Ue(y.width - F, we[0])), S(g, y.top + Ue(y.height - $, we[1] || we[0])), S(l, F), S(c, $), S(w, F > 0 && $ > 0, !0);
  }
  function dt(f, y, T, A) {
    return r(this, void 0, void 0, function* () {
      if (!(!f.complete || f.naturalWidth <= 0)) {
        try {
          yield f.decode();
        } catch {
          if (!f.complete || f.naturalWidth <= 0) return;
        }
        A !== Pe || y !== M || f !== N || T !== be(f) || (Q = y, ge = T, j(), Se = "", se());
      }
    });
  }
  function Bt(f) {
    const y = be(f);
    if (!y) {
      S(w, !1);
      return;
    }
    if (Se && y === Se) {
      S(w, !1);
      return;
    }
    j(), Se = "";
    const T = ++Pe;
    dt(f, M, y, T);
  }
  function Nt() {
    N && Bt(N);
  }
  function pt() {
    if (!J) return;
    const y = Array.from(J.querySelectorAll("img")).sort((T, A) => {
      const L = T.getBoundingClientRect(), F = A.getBoundingClientRect();
      return F.width * F.height - L.width * L.height;
    })[0] || null;
    if (y !== N && (N?.removeEventListener("load", Nt), N = y, N?.addEventListener("load", Nt), ne?.disconnect(), ne = new ResizeObserver(se), ne.observe(J), N && ne.observe(N)), !N) {
      S(w, !1), Q = "", ge = "";
      return;
    }
    Bt(N);
  }
  function Ot() {
    ze();
    const f = z();
    if (f) {
      if (J = document.getElementById(f), !J) {
        Re = setTimeout(Ot, 100);
        return;
      }
      he = new MutationObserver(pt), he.observe(J, {
        childList: !0,
        subtree: !0,
        attributes: !0,
        attributeFilter: ["src", "srcset", "style", "class"]
      }), pt();
    }
  }
  Pa(() => {
    var f, y;
    return re = !0, xe = z(), Ot(), window.addEventListener("resize", se), window.addEventListener("scroll", se, !0), (f = window.visualViewport) === null || f === void 0 || f.addEventListener("resize", se), (y = window.visualViewport) === null || y === void 0 || y.addEventListener("scroll", se), () => {
      var T, A;
      re = !1, window.removeEventListener("resize", se), window.removeEventListener("scroll", se, !0), (T = window.visualViewport) === null || T === void 0 || T.removeEventListener("resize", se), (A = window.visualViewport) === null || A === void 0 || A.removeEventListener("scroll", se), ze();
    };
  });
  function Wt() {
    if (z()) return {
      left: 0,
      top: 0,
      width: h(l),
      height: h(c)
    };
    const f = ce(), y = Fe();
    if (f <= 0 || y <= 0) return { left: 0, top: 0, width: 0, height: 0 };
    const T = Math.min(h(l) / f, h(c) / y), A = f * T, L = y * T;
    return {
      left: (h(l) - A) / 2,
      top: (h(c) - L) / 2,
      width: A,
      height: L
    };
  }
  function Zt(f, y, T) {
    return Math.max(y, Math.min(T, f));
  }
  function Lt(f, y) {
    const T = s.getBoundingClientRect(), A = Wt(), L = f.clientX - T.left, F = f.clientY - T.top;
    return !(L >= A.left && L <= A.left + A.width && F >= A.top && F <= A.top + A.height) && !y || A.width <= 0 || A.height <= 0 ? null : {
      x: Zt((L - A.left) / A.width * ce(), 0, ce()),
      y: Zt((F - A.top) / A.height * Fe(), 0, Fe())
    };
  }
  function G(f) {
    return [Number(f.x.toFixed(4)), Number(f.y.toFixed(4))];
  }
  function q(f, y, T) {
    const A = ee(), L = {
      gesture: f,
      start_xy: G(y),
      end_xy: G(T),
      expected_revision: Number.isInteger(A.revision) ? Number(A.revision) : null,
      image_id: String(A.image_id || ""),
      image_sha256: String(A.image_sha256 || "")
    };
    S(u, Object.assign(Object.assign({}, h(u)), { client_intent: L }), !0);
    const F = { client_intent: Object.assign({}, L) };
    i.props.value = F, O = JSON.stringify(F), i.dispatch("input");
  }
  function He(f) {
    if (f.button !== 0 || !Pt() || v !== null) return;
    const y = Lt(f, !1);
    y && (f.preventDefault(), _ = h(E) ? Object.assign({}, h(E)) : null, B = h(b) ? Object.assign({}, h(b)) : null, S(d, y, !0), S(m, y, !0), I = { x: f.clientX, y: f.clientY }, H = 0, v = f.pointerId, x = f.pointerType || "mouse", s.setPointerCapture(f.pointerId));
  }
  function nt(f) {
    if (f.pointerId !== v || !h(d) || !I) return;
    f.preventDefault();
    const y = Lt(f, !0);
    y && (H = Math.max(H, Math.hypot(f.clientX - I.x, f.clientY - I.y)), fe() && S(m, y, !0));
  }
  function _e() {
    S(E, _, !0), S(b, B, !0);
  }
  function it(f) {
    const y = v;
    v = null, y !== null && s?.hasPointerCapture(y) && s.releasePointerCapture(y), x = "mouse", I = null, S(d, null), S(m, null), _ = null, B = null, H = 0, f && f.preventDefault();
  }
  function Mt(f) {
    if (f.pointerId !== v || !h(d) || !I) return;
    const y = Object.assign({}, h(d)), T = Lt(f, !0) || Object.assign({}, h(m));
    H = Math.max(H, Math.hypot(f.clientX - I.x, f.clientY - I.y));
    const A = x;
    let L = null, F = y, $ = T;
    ht() && H <= je(A) ? (L = "click", F = T, $ = T, _e()) : fe() && H >= rt(A) && Math.abs(T.x - y.x) >= a && Math.abs(T.y - y.y) >= a ? (L = "drag", S(E, y, !0), S(b, T, !0)) : _e(), it(f), L && q(L, F, $);
  }
  function Je() {
    v !== null && (_e(), it());
  }
  function Xe(f) {
    f.pointerId === v && Je();
  }
  function at(f) {
    f.pointerId === v && Je();
  }
  function vt() {
    Je();
  }
  function mt() {
    const f = h(d) && fe() ? h(d) : h(E), y = h(m) && fe() ? h(m) : h(b);
    if (!f || !y) return null;
    const T = Wt();
    if (T.width <= 0 || T.height <= 0) return null;
    const A = T.left + Math.min(f.x, y.x) / ce() * T.width, L = T.top + Math.min(f.y, y.y) / Fe() * T.height, F = Math.abs(y.x - f.x) / ce() * T.width, $ = Math.abs(y.y - f.y) / Fe() * T.height;
    return `left:${A}px;top:${L}px;width:${F}px;height:${$}px`;
  }
  function qe() {
    return h(d) !== null || Ve() !== "applied";
  }
  function Qe() {
    return Pt() ? fe() ? "crosshair" : "pointer" : "default";
  }
  function Ne() {
    const f = Qe();
    return z() ? `position:fixed;left:${h(p)}px;top:${h(g)}px;width:${h(l)}px;height:${h(c)}px;cursor:${f}` : `position:relative;height:${ke(i.props.height)};cursor:${f}`;
  }
  st("blur", ma, vt), is(e, {
    get visible() {
      return i.shared.visible;
    },
    variant: "solid",
    border_mode: "none",
    padding: !1,
    get elem_id() {
      return i.shared.elem_id;
    },
    get elem_classes() {
      return i.shared.elem_classes;
    },
    allow_overflow: !0,
    get container() {
      return i.shared.container;
    },
    get scale() {
      return i.shared.scale;
    },
    get min_width() {
      return i.shared.min_width;
    },
    children: (f, y) => {
      var T = hl(), A = ve(T);
      al(A, ts(
        {
          get autoscroll() {
            return i.shared.autoscroll;
          },
          get i18n() {
            return i.i18n;
          }
        },
        () => i.shared.loading_status,
        {
          on_clear_status: () => i.dispatch("clear_status", i.shared.loading_status)
        }
      ));
      var L = W(A, 2);
      let F;
      var $ = le(L);
      {
        var we = (de) => {
          var pe = cl();
          let Ke;
          Y(
            (oe, Oe) => {
              Ke = Ye(pe, 1, "selection-rectangle svelte-r41nsf", null, Ke, oe), Ce(pe, Oe);
            },
            [
              () => ({
                "draft-selection": qe(),
                "applied-selection": !qe()
              }),
              () => mt()
            ]
          ), C(de, pe);
        }, gt = Le(() => mt());
        Z($, (de) => {
          h(gt) && de(we);
        });
      }
      tn(L, (de) => s = de, () => s), Fa(L, (de) => Ee?.(de)), We(() => st("pointerdown", L, He)), We(() => st("pointermove", L, nt)), We(() => st("pointerup", L, Mt)), We(() => st("pointercancel", L, Xe)), We(() => st("lostpointercapture", L, at)), Y(
        (de, pe) => {
          F = Ye(L, 1, "gesture-surface svelte-r41nsf", null, F, de), Ce(L, pe);
        },
        [() => ({ disabled: !Pt() }), () => Ne()]
      ), C(f, T);
    },
    $$slots: { default: !0 }
  }), ur();
}
export {
  pl as default
};

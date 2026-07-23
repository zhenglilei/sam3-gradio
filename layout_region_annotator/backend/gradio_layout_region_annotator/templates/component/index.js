import { i as Dr, g as Ln, o as Ai, n as Xe, u as ae, s as Hi, r as yr, m as Ye, a as E, b as f, t as kr, d as Pi, q as Ii, c as Nn, e as Qe, f as qt, h as jt, j as Bi, T as Oi, k as Li, l as Vt, p as Je, v as Ur, w as Ke, x as Mn, y as Cn, z as Rn, A as St, E as Wt, B as Kr, C as Ni, D as Dn, F as Fr, G as Mi, H as $r, I as Ci, J as Ri, K as Re, L as kn, M as or, N as Di, O as ki, P as Ui, Q as Fi, R as Un, S as Gr, U as en, V as tn, W as Gi, X as ji, Y as Vi, Z as zi, _ as Xi, $ as qi, a0 as Wi, a1 as Zi, a2 as jr, a3 as Yi, a4 as Fn, a5 as Zt, a6 as Ji, a7 as Qi, a8 as Ki, a9 as $i, aa as Gn, ab as ea, ac as ta, ad as Vr, ae as ra, af as Te, ag as na, ah as xe, ai as xr, aj as Er, ak as ia, al as aa, am as wt, an as sa, ao as oa, ap as la, aq as ua, ar as fa, as as ca, at as jn, au as bt, av as ha, aw as rn, ax as da, ay as fe, az as Yt, aA as Jt, aB as j, aC as Oe, aD as Q, aE as pa, aF as te, aG as ue, aH as we, aI as q, aJ as va, aK as ma } from "./render-DNiZdw6o.js";
const ga = [];
function ba(e, t = !1, r = !1) {
  return Ut(e, /* @__PURE__ */ new Map(), "", ga, null, r);
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
    if (Dr(e)) {
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
    if (Ln(e) === Ai) {
      s = {}, t.set(e, s), i !== null && t.set(i, s);
      for (var c of Object.keys(e))
        s[c] = Ut(
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
function zr(e, t, r) {
  if (e == null)
    return t(void 0), r && r(void 0), Xe;
  const n = ae(
    () => e.subscribe(
      t,
      // @ts-expect-error
      r
    )
  );
  return n.unsubscribe ? () => n.unsubscribe() : n;
}
const rt = [];
function _a(e, t) {
  return {
    subscribe: At(e, t).subscribe
  };
}
function At(e, t = Xe) {
  let r = null;
  const n = /* @__PURE__ */ new Set();
  function i(s) {
    if (Hi(e, s) && (e = s, r)) {
      const u = !rt.length;
      for (const l of n)
        l[1](), rt.push(l, e);
      if (u) {
        for (let l = 0; l < rt.length; l += 2)
          rt[l][0](rt[l + 1]);
        rt.length = 0;
      }
    }
  }
  function a(s) {
    i(s(
      /** @type {T} */
      e
    ));
  }
  function o(s, u = Xe) {
    const l = [s, u];
    return n.add(l), n.size === 1 && (r = t(i, a) || Xe), s(
      /** @type {T} */
      e
    ), () => {
      n.delete(l), n.size === 0 && r && (r(), r = null);
    };
  }
  return { set: i, update: a, subscribe: o };
}
function ft(e, t, r) {
  const n = !Array.isArray(e), i = n ? [e] : e;
  if (!i.every(Boolean))
    throw new Error("derived() expects stores as input, got a falsy value");
  const a = t.length < 2;
  return _a(r, (o, s) => {
    let u = !1;
    const l = [];
    let c = 0, d = Xe;
    const v = () => {
      if (c)
        return;
      d();
      const g = t(n ? l[0] : l, o, s);
      a ? o(g) : d = typeof g == "function" ? g : Xe;
    }, w = i.map(
      (g, x) => zr(
        g,
        (H) => {
          l[x] = H, c &= ~(1 << x), u && v();
        },
        () => {
          c |= 1 << x;
        }
      )
    );
    return u = !0, v(), function() {
      yr(w), d(), u = !1;
    };
  });
}
function ya(e) {
  let t;
  return zr(e, (r) => t = r)(), t;
}
let Rt = !1, wr = /* @__PURE__ */ Symbol("unmounted");
function nn(e, t, r) {
  const n = r[t] ??= {
    store: null,
    source: Ye(void 0),
    unsubscribe: Xe
  };
  if (n.store !== e && !(wr in r))
    if (n.unsubscribe(), n.store = e ?? null, e == null)
      n.source.v = void 0, n.unsubscribe = Xe;
    else {
      var i = !0;
      n.unsubscribe = zr(e, (a) => {
        i ? n.source.v = a : E(n.source, a);
      }), i = !1;
    }
  return e && wr in r ? ya(e) : f(n.source);
}
function xa() {
  const e = {};
  function t() {
    kr(() => {
      for (var r in e)
        e[r].unsubscribe();
      Pi(e, wr, {
        enumerable: !1,
        value: !0
      });
    });
  }
  return [e, t];
}
function Ea(e) {
  var t = Rt;
  try {
    return Rt = !1, [e(), Rt];
  } finally {
    Rt = t;
  }
}
function wa(e, t) {
  if (t) {
    const r = document.body;
    e.autofocus = !0, Ii(() => {
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
function Vn(e) {
  var t = Nn("template");
  return t.innerHTML = Sa(e.replaceAll("<!>", "<!---->")), t.content;
}
function st(e, t) {
  var r = (
    /** @type {Effect} */
    qt
  );
  r.nodes === null && (r.nodes = { start: e, end: t, a: null, t: null });
}
// @__NO_SIDE_EFFECTS__
function se(e, t) {
  var r = (t & Oi) !== 0, n = (t & Li) !== 0, i, a = !e.startsWith("<!>");
  return () => {
    i === void 0 && (i = Vn(a ? e : "<!>" + e), r || (i = /** @type {TemplateNode} */
    jt(i)));
    var o = (
      /** @type {TemplateNode} */
      n || Bi ? document.importNode(i, !0) : i.cloneNode(!0)
    );
    if (r) {
      var s = (
        /** @type {TemplateNode} */
        jt(o)
      ), u = (
        /** @type {TemplateNode} */
        o.lastChild
      );
      st(s, u);
    } else
      st(o, o);
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
        Vn(i)
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
    return st(u, u), u;
  };
}
// @__NO_SIDE_EFFECTS__
function zn(e, t) {
  return /* @__PURE__ */ Aa(e, t, "svg");
}
function Be(e = "") {
  {
    var t = Qe(e + "");
    return st(t, t), t;
  }
}
function it() {
  var e = document.createDocumentFragment(), t = document.createComment(""), r = Qe();
  return e.append(t, r), st(t, r), e;
}
function C(e, t) {
  e !== null && e.before(
    /** @type {Node} */
    t
  );
}
class Qt {
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
        s && (Je(s.effect), this.#e.delete(o));
      }
      for (const [a, o] of this.#r) {
        if (a === r || this.#n.has(a)) continue;
        const s = () => {
          if (Array.from(this.#t.values()).includes(a)) {
            var l = document.createDocumentFragment();
            Cn(o, l), l.append(Qe()), this.#e.set(a, { effect: o, fragment: l });
          } else
            Je(o);
          this.#n.delete(a), this.#r.delete(a);
        };
        this.#i || !n ? (this.#n.add(a), Ur(o, s, !1)) : s();
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
      r.includes(n) || (Je(i.effect), this.#e.delete(n));
  };
  /**
   *
   * @param {any} key
   * @param {null | ((target: TemplateNode) => void)} fn
   */
  ensure(t, r) {
    var n = (
      /** @type {Batch} */
      Mn
    ), i = Rn();
    if (r && !this.#r.has(t) && !this.#e.has(t))
      if (i) {
        var a = document.createDocumentFragment(), o = Qe();
        a.append(o), this.#e.set(t, {
          effect: Ke(() => r(o)),
          fragment: a
        });
      } else
        this.#r.set(
          t,
          Ke(() => r(this.anchor))
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
function Ha(e, t, ...r) {
  var n = new Qt(e);
  St(() => {
    const i = t() ?? null;
    n.ensure(i, i && ((a) => i(a, ...r)));
  }, Wt);
}
function Y(e, t, r = !1) {
  var n = new Qt(e), i = r ? Wt : 0;
  function a(o, s) {
    n.ensure(o, s);
  }
  St(() => {
    var o = !1;
    t((s, u = 0) => {
      o = !0, a(u, s);
    }), o || a(-1, null);
  }, i);
}
function an(e, t) {
  return t;
}
function Pa(e, t, r) {
  for (var n = [], i = t.length, a, o = t.length, s = 0; s < i; s++) {
    let d = t[s];
    Ur(
      d,
      () => {
        if (a) {
          if (a.pending.delete(d), a.done.add(d), a.pending.size === 0) {
            var v = (
              /** @type {Set<EachOutroGroup>} */
              e.outrogroups
            );
            Tr(e, Fr(a.done)), v.delete(a), v.size === 0 && (e.outrogroups = null);
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
      ki(c), c.append(l), e.items.clear();
    }
    Tr(e, t, !u);
  } else
    a = {
      pending: new Set(t),
      done: /* @__PURE__ */ new Set()
    }, (e.outrogroups ??= /* @__PURE__ */ new Set()).add(a);
}
function Tr(e, t, r = !0) {
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
      a.f |= Re;
      const o = document.createDocumentFragment();
      Cn(a, o);
    } else
      Je(t[i], r);
  }
}
var sn;
function on(e, t, r, n, i, a = null) {
  var o = e, s = /* @__PURE__ */ new Map(), u = null, l = Dn(() => {
    var h = r();
    return (
      /** @type {V[]} */
      Dr(h) ? h : h == null ? [] : Fr(h)
    );
  }), c, d = /* @__PURE__ */ new Map(), v = !0;
  function w(h) {
    (H.effect.f & kn) === 0 && (H.pending.delete(h), H.fallback = u, Ia(H, c, o, t, n), u !== null && (c.length === 0 ? (u.f & Re) === 0 ? Vt(u) : (u.f ^= Re, xt(u, null, o)) : Ur(u, () => {
      u = null;
    })));
  }
  function g(h) {
    H.pending.delete(h);
  }
  var x = St(() => {
    c = /** @type {V[]} */
    f(l);
    for (var h = c.length, m = /* @__PURE__ */ new Set(), p = (
      /** @type {Batch} */
      Mn
    ), b = Rn(), y = 0; y < h; y += 1) {
      var A = c[y], P = n(A, y), B = v ? null : s.get(P);
      B ? (B.v && Kr(B.v, A), B.i && Kr(B.i, y), b && p.unskip_effect(B.e)) : (B = Ba(
        s,
        v ? o : sn ??= Qe(),
        A,
        P,
        y,
        i,
        t,
        r
      ), v || (B.e.f |= Re), s.set(P, B)), m.add(P);
    }
    if (h === 0 && a && !u && (v ? u = Ke(() => a(o)) : (u = Ke(() => a(sn ??= Qe())), u.f |= Re)), h > m.size && Ni(), !v)
      if (d.set(p, m), b) {
        for (const [D, L] of s)
          m.has(D) || p.skip_effect(L.e);
        p.oncommit(w), p.ondiscard(g);
      } else
        w(p);
    f(l);
  }), H = { effect: x, items: s, pending: d, outrogroups: null, fallback: u };
  v = !1;
}
function _t(e) {
  for (; e !== null && (e.f & Di) === 0; )
    e = e.next;
  return e;
}
function Ia(e, t, r, n, i) {
  var a = t.length, o = e.items, s = _t(e.effect.first), u, l = null, c = [], d = [], v, w, g, x;
  for (x = 0; x < a; x += 1) {
    if (v = t[x], w = i(v, x), g = /** @type {EachItem} */
    o.get(w).e, e.outrogroups !== null)
      for (const B of e.outrogroups)
        B.pending.delete(g), B.done.delete(g);
    if ((g.f & or) !== 0 && Vt(g), (g.f & Re) !== 0)
      if (g.f ^= Re, g === s)
        xt(g, null, r);
      else {
        var H = l ? l.next : s;
        g === e.effect.last && (e.effect.last = g.prev), g.prev && (g.prev.next = g.next), g.next && (g.next.prev = g.prev), Ve(e, l, g), Ve(e, g, H), xt(g, H, r), l = g, c = [], d = [], s = _t(l.next);
        continue;
      }
    if (g !== s) {
      if (u !== void 0 && u.has(g)) {
        if (c.length < d.length) {
          var h = d[0], m;
          l = h.prev;
          var p = c[0], b = c[c.length - 1];
          for (m = 0; m < c.length; m += 1)
            xt(c[m], h, r);
          for (m = 0; m < d.length; m += 1)
            u.delete(d[m]);
          Ve(e, p.prev, b.next), Ve(e, l, p), Ve(e, b, h), s = h, l = b, x -= 1, c = [], d = [];
        } else
          u.delete(g), xt(g, s, r), Ve(e, g.prev, g.next), Ve(e, g, l === null ? e.effect.first : l.next), Ve(e, l, g), l = g;
        continue;
      }
      for (c = [], d = []; s !== null && s !== g; )
        (u ??= /* @__PURE__ */ new Set()).add(s), d.push(s), s = _t(s.next);
      if (s === null)
        continue;
    }
    (g.f & Re) === 0 && c.push(g), l = g, s = _t(g.next);
  }
  if (e.outrogroups !== null) {
    for (const B of e.outrogroups)
      B.pending.size === 0 && (Tr(e, Fr(B.done)), e.outrogroups?.delete(B));
    e.outrogroups.size === 0 && (e.outrogroups = null);
  }
  if (s !== null || u !== void 0) {
    var y = [];
    if (u !== void 0)
      for (g of u)
        (g.f & or) === 0 && y.push(g);
    for (; s !== null; )
      (s.f & or) === 0 && s !== e.fallback && y.push(s), s = _t(s.next);
    var A = y.length;
    if (A > 0) {
      var P = null;
      Pa(e, y, P);
    }
  }
}
function Ba(e, t, r, n, i, a, o, s) {
  var u = (o & Ci) !== 0 ? (o & Ri) === 0 ? Ye(r, !1, !1) : $r(r) : null, l = (o & Mi) !== 0 ? $r(i) : null;
  return {
    v: u,
    i: l,
    e: Ke(() => (a(t, u ?? r, l ?? i, s), () => {
      e.delete(n);
    }))
  };
}
function xt(e, t, r) {
  if (e.nodes)
    for (var n = e.nodes.start, i = e.nodes.end, a = t && (t.f & Re) === 0 ? (
      /** @type {EffectNodes} */
      t.nodes.start
    ) : r; n !== null; ) {
      var o = (
        /** @type {TemplateNode} */
        Ui(n)
      );
      if (a.before(n), n === i)
        return;
      n = o;
    }
}
function Ve(e, t, r) {
  t === null ? e.effect.first = r : t.next = r, r === null ? e.effect.last = t : r.prev = t;
}
function Sr(e, t, r, n, i) {
  var a = t.$$slots?.[r], o = !1;
  a === !0 && (a = t[r === "default" ? "children" : r], o = !0), a === void 0 || a(e, o ? () => n : n);
}
function Oa(e, t, r) {
  var n = new Qt(e);
  St(() => {
    var i = t() ?? null;
    n.ensure(i, i && ((a) => r(a, i)));
  }, Wt);
}
const La = () => performance.now(), Se = {
  // don't access requestAnimationFrame eagerly outside method
  // this allows basic testing of user code without JSDOM
  // bunder will eval and remove ternary when the user's app is built
  tick: (
    /** @param {any} _ */
    (e) => requestAnimationFrame(e)
  ),
  now: () => La(),
  tasks: /* @__PURE__ */ new Set()
};
function Xn() {
  const e = Se.now();
  Se.tasks.forEach((t) => {
    t.c(e) || (Se.tasks.delete(t), t.f());
  }), Se.tasks.size !== 0 && Se.tick(Xn);
}
function Na(e) {
  let t;
  return Se.tasks.size === 0 && Se.tick(Xn), {
    promise: new Promise((r) => {
      Se.tasks.add(t = { c: e, f: r });
    }),
    abort() {
      Se.tasks.delete(t);
    }
  };
}
function Ma(e, t, r, n, i, a) {
  var o = null, s = (
    /** @type {TemplateNode} */
    e
  ), u = new Qt(s, !1);
  St(() => {
    const l = t() || null;
    var c = l === "svg" ? Fi : void 0;
    if (l === null) {
      u.ensure(null, null);
      return;
    }
    return u.ensure(l, (d) => {
      if (l) {
        if (o = Nn(l, c), st(o, o), n) {
          var v = null, w = o.appendChild(Qe());
          n(o, w), v?.remove();
        }
        qt.nodes.end = o, d.before(o);
      }
    }), () => {
    };
  }, Wt), kr(() => {
  });
}
function Ca(e, t) {
  var r = void 0, n;
  Un(() => {
    r !== (r = t()) && (n && (Je(n), n = null), r && (n = Ke(() => {
      Gr(() => (
        /** @type {(node: Element) => void} */
        r(e)
      ));
    })));
  });
}
function qn(e) {
  var t, r, n = "";
  if (typeof e == "string" || typeof e == "number") n += e;
  else if (typeof e == "object") if (Array.isArray(e)) {
    var i = e.length;
    for (t = 0; t < i; t++) e[t] && (r = qn(e[t])) && (n && (n += " "), n += r);
  } else for (r in e) e[r] && (n && (n += " "), n += r);
  return n;
}
function Ra() {
  for (var e, t, r = 0, n = "", i = arguments.length; r < i; r++) (e = arguments[r]) && (t = qn(e)) && (n && (n += " "), n += t);
  return n;
}
function Da(e) {
  return typeof e == "object" ? Ra(e) : e ?? "";
}
const ln = [...` \t
\r\f \v\uFEFF`];
function ka(e, t, r) {
  var n = e == null ? "" : "" + e;
  if (t && (n = n ? n + " " + t : t), r) {
    for (var i of Object.keys(r))
      if (r[i])
        n = n ? n + " " + i : i;
      else if (n.length)
        for (var a = i.length, o = 0; (o = n.indexOf(i, o)) >= 0; ) {
          var s = o + a;
          (o === 0 || ln.includes(n[o - 1])) && (s === n.length || ln.includes(n[s])) ? n = (o === 0 ? "" : n.substring(0, o)) + n.substring(s + 1) : o = s;
        }
  }
  return n === "" ? null : n;
}
function un(e, t = !1) {
  var r = t ? " !important;" : ";", n = "";
  for (var i of Object.keys(e)) {
    var a = e[i];
    a != null && a !== "" && (n += " " + i + ": " + a + r);
  }
  return n;
}
function lr(e) {
  return e[0] !== "-" || e[1] !== "-" ? e.toLowerCase() : e;
}
function Ua(e, t) {
  if (t) {
    var r = "", n, i;
    if (Array.isArray(t) ? (n = t[0], i = t[1]) : n = t, e) {
      e = String(e).replaceAll(/\s*\/\*.*?\*\/\s*/g, "").trim();
      var a = !1, o = 0, s = !1, u = [];
      n && u.push(...Object.keys(n).map(lr)), i && u.push(...Object.keys(i).map(lr));
      var l = 0, c = -1;
      const x = e.length;
      for (var d = 0; d < x; d++) {
        var v = e[d];
        if (s ? v === "/" && e[d - 1] === "*" && (s = !1) : a ? a === v && (a = !1) : v === "/" && e[d + 1] === "*" ? s = !0 : v === '"' || v === "'" ? a = v : v === "(" ? o++ : v === ")" && o--, !s && a === !1 && o === 0) {
          if (v === ":" && c === -1)
            c = d;
          else if (v === ";" || d === x - 1) {
            if (c !== -1) {
              var w = lr(e.substring(l, c).trim());
              if (!u.includes(w)) {
                v !== ";" && d++;
                var g = e.substring(l, d).trim();
                r += " " + g + ";";
              }
            }
            l = d + 1, c = -1;
          }
        }
      }
    }
    return n && (r += un(n)), i && (r += un(i, !0)), r = r.trim(), r === "" ? null : r;
  }
  return e == null ? null : String(e);
}
function De(e, t, r, n, i, a) {
  var o = (
    /** @type {any} */
    e[en]
  );
  if (o !== r || o === void 0) {
    var s = ka(r, n, a);
    s == null ? e.removeAttribute("class") : t ? e.className = s : e.setAttribute("class", s), e[en] = r;
  } else if (a && i !== a)
    for (var u in a) {
      var l = !!a[u];
      (i == null || l !== !!i[u]) && e.classList.toggle(u, l);
    }
  return a;
}
function ur(e, t = {}, r, n) {
  for (var i in r) {
    var a = r[i];
    t[i] !== a && (r[i] == null ? e.style.removeProperty(i) : e.style.setProperty(i, a, n));
  }
}
function Ae(e, t, r, n) {
  var i = (
    /** @type {any} */
    e[tn]
  );
  if (i !== t) {
    var a = Ua(t, n);
    a == null ? e.removeAttribute("style") : e.style.cssText = a, e[tn] = t;
  } else n && (Array.isArray(n) ? (ur(e, r?.[0], n[0]), ur(e, r?.[1], n[1], "important")) : ur(e, r, n));
  return n;
}
function Ar(e, t, r = !1) {
  if (e.multiple) {
    if (t == null)
      return;
    if (!Dr(t))
      return Gi();
    for (var n of e.options)
      n.selected = t.includes(fn(n));
    return;
  }
  for (n of e.options) {
    var i = fn(n);
    if (ji(i, t)) {
      n.selected = !0;
      return;
    }
  }
  (!r || t !== void 0) && (e.selectedIndex = -1);
}
function Fa(e) {
  var t = new MutationObserver(() => {
    Ar(e, e.__value);
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
  }), kr(() => {
    t.disconnect();
  });
}
function fn(e) {
  return "__value" in e ? e.__value : e.value;
}
const Et = /* @__PURE__ */ Symbol("class"), nt = /* @__PURE__ */ Symbol("style"), Wn = /* @__PURE__ */ Symbol("is custom element"), Zn = /* @__PURE__ */ Symbol("is html"), Ga = jr ? "input" : "INPUT", ja = jr ? "option" : "OPTION", Va = jr ? "select" : "SELECT";
function za(e, t) {
  t ? e.hasAttribute("selected") || e.setAttribute("selected", "") : e.removeAttribute("selected");
}
function at(e, t, r, n) {
  var i = Yn(e);
  i[t] !== (i[t] = r) && (t === "loading" && (e[Vi] = r), r == null ? e.removeAttribute(t) : typeof r != "string" && Jn(e).includes(t) ? e[t] = r : e.setAttribute(t, r));
}
function Xa(e, t, r, n, i = !1, a = !1) {
  var o = Yn(e), s = o[Wn], u = !o[Zn], l = t || {}, c = e.nodeName === ja;
  for (var d in t)
    d in r || (r[d] = null);
  r.class ? r.class = Da(r.class) : r.class = null, r[nt] && (r.style ??= null);
  var v = Jn(e);
  if (e.nodeName === Ga && "type" in r && ("value" in r || "__value" in r)) {
    var w = r.type;
    (w !== l.type || w === void 0 && e.hasAttribute("type")) && (l.type = w, at(e, "type", w));
  }
  for (const b in r) {
    let y = r[b];
    if (c && b === "value" && y == null) {
      e.value = e.__value = "", l[b] = y;
      continue;
    }
    if (b === "class") {
      var g = e.namespaceURI === "http://www.w3.org/1999/xhtml";
      De(e, g, y, n, t?.[Et], r[Et]), l[b] = y, l[Et] = r[Et];
      continue;
    }
    if (b === "style") {
      Ae(e, y, t?.[nt], r[nt]), l[b] = y, l[nt] = r[nt];
      continue;
    }
    var x = l[b];
    if (!(y === x && !(y === void 0 && e.hasAttribute(b)))) {
      l[b] = y;
      var H = b[0] + b[1];
      if (H !== "$$")
        if (H === "on") {
          const A = {}, P = "$$" + b;
          let B = b.slice(2);
          var h = $i(B);
          if (Yi(B) && (B = B.slice(0, -7), A.capture = !0), !h && x) {
            if (y != null) continue;
            e.removeEventListener(B, l[P], A), l[P] = null;
          }
          if (h)
            Fn(B, e, y), Zt([B]);
          else if (y != null) {
            let D = function(L) {
              l[b].call(this, L);
            };
            l[P] = Ji(B, e, D, A);
          }
        } else if (b === "style")
          at(e, b, y);
        else if (b === "autofocus")
          wa(
            /** @type {HTMLElement} */
            e,
            !!y
          );
        else if (!s && (b === "__value" || b === "value" && y != null))
          e.value = e.__value = y;
        else if (b === "selected" && c)
          za(
            /** @type {HTMLOptionElement} */
            e,
            y
          );
        else {
          var m = b;
          u || (m = Qi(m));
          var p = m === "defaultValue" || m === "defaultChecked";
          if (y == null && !s && !p)
            if (o[b] = null, m === "value" || m === "checked") {
              let A = (
                /** @type {HTMLInputElement} */
                e
              );
              const P = t === void 0;
              if (m === "value") {
                let B = A.defaultValue;
                A.removeAttribute(m), A.defaultValue = B, A.value = A.__value = P ? B : null;
              } else {
                let B = A.defaultChecked;
                A.removeAttribute(m), A.defaultChecked = B, A.checked = P ? B : !1;
              }
            } else
              e.removeAttribute(b);
          else p || v.includes(m) && (s || typeof y != "string") ? (e[m] = y, m in o && (o[m] = Ki)) : typeof y != "function" && at(e, m, y);
        }
    }
  }
  return l;
}
function qa(e, t, r = [], n = [], i = [], a, o = !1, s = !1) {
  Wi(i, r, n, (u) => {
    var l = void 0, c = {}, d = e.nodeName === Va, v = !1;
    if (Un(() => {
      var g = t(...u.map(f)), x = Xa(
        e,
        l,
        g,
        a,
        o,
        s
      );
      v && d && "value" in g && Ar(
        /** @type {HTMLSelectElement} */
        e,
        g.value
      );
      for (let h of Object.getOwnPropertySymbols(c))
        g[h] || Je(c[h]);
      for (let h of Object.getOwnPropertySymbols(g)) {
        var H = g[h];
        h.description === Zi && (!l || H !== l[h]) && (c[h] && Je(c[h]), c[h] = Ke(() => Ca(e, () => H))), x[h] = H;
      }
      l = x;
    }), d) {
      var w = (
        /** @type {HTMLSelectElement} */
        e
      );
      Gr(() => {
        Ar(
          w,
          /** @type {Record<string | symbol, any>} */
          l.value,
          !0
        ), Fa(w);
      });
    }
    v = !0;
  });
}
function Yn(e) {
  return (
    /** @type {Record<string | symbol, unknown>} **/
    /** @type {any} */
    e[zi] ??= {
      [Wn]: e.nodeName.includes("-"),
      [Zn]: e.namespaceURI === Xi
    }
  );
}
var cn = /* @__PURE__ */ new Map();
function Jn(e) {
  var t = e.getAttribute("is") || e.nodeName, r = cn.get(t);
  if (r) return r;
  cn.set(t, r = []);
  for (var n, i = e, a = Element.prototype; a !== i; ) {
    n = qi(i);
    for (var o in n)
      n[o].set && // better safe than sorry, we don't want spread attributes to mess with HTML content
      o !== "innerHTML" && o !== "textContent" && o !== "innerText" && r.push(o);
    i = Ln(i);
  }
  return r;
}
function fr(e, t) {
  return e === t || e?.[Vr] === t;
}
function Xr(e = {}, t, r, n) {
  var i = (
    /** @type {ComponentContext} */
    Gn.r
  ), a = (
    /** @type {Effect} */
    qt
  );
  return Gr(() => {
    var o, s;
    return ea(() => {
      o = s, s = [], ae(() => {
        fr(r(...s), e) || (t(e, ...s), o && fr(r(...o), e) && t(null, ...o));
      });
    }), () => {
      let u = a;
      for (; u !== i && u.parent !== null && u.parent.f & ta; )
        u = u.parent;
      const l = () => {
        s && fr(r(...s), e) && t(null, ...s);
      }, c = u.teardown;
      u.teardown = () => {
        l(), c?.();
      };
    };
  }), e;
}
function Wa(e = !1) {
  const t = (
    /** @type {ComponentContextLegacy} */
    Gn
  ), r = t.l.u;
  if (!r) return;
  let n = () => xe(t.s);
  if (e) {
    let i = 0, a = (
      /** @type {Record<string, any>} */
      {}
    );
    const o = xr(() => {
      let s = !1;
      const u = t.s;
      for (const l in u)
        u[l] !== a[l] && (a[l] = u[l], s = !0);
      return s && i++, i;
    });
    n = () => f(o);
  }
  r.b.length && ra(() => {
    hn(t, n), yr(r.b);
  }), Te(() => {
    const i = ae(() => r.m.map(na));
    return () => {
      for (const a of i)
        typeof a == "function" && a();
    };
  }), r.a.length && Te(() => {
    hn(t, n), yr(r.a);
  });
}
function hn(e, t) {
  if (e.l.s)
    for (const r of e.l.s) f(r);
  t();
}
const Za = {
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
function Ya(e, t, r) {
  return new Proxy(
    { props: e, exclude: t },
    Za
  );
}
const Ja = {
  get(e, t) {
    let r = e.props.length;
    for (; r--; ) {
      let n = e.props[r];
      if (bt(n) && (n = n()), typeof n == "object" && n !== null && t in n) return n[t];
    }
  },
  set(e, t, r) {
    let n = e.props.length;
    for (; n--; ) {
      let i = e.props[n];
      bt(i) && (i = i());
      const a = Er(i, t);
      if (a && a.set)
        return a.set(r), !0;
    }
    return !1;
  },
  getOwnPropertyDescriptor(e, t) {
    let r = e.props.length;
    for (; r--; ) {
      let n = e.props[r];
      if (bt(n) && (n = n()), typeof n == "object" && n !== null && t in n) {
        const i = Er(n, t);
        return i && !i.configurable && (i.configurable = !0), i;
      }
    }
  },
  has(e, t) {
    if (t === Vr || t === jn) return !1;
    for (let r of e.props)
      if (bt(r) && (r = r()), r != null && t in r) return !0;
    return !1;
  },
  ownKeys(e) {
    const t = [];
    for (let r of e.props)
      if (bt(r) && (r = r()), !!r) {
        for (const n in r)
          t.includes(n) || t.push(n);
        for (const n of Object.getOwnPropertySymbols(r))
          t.includes(n) || t.push(n);
      }
    return t;
  }
};
function Qa(...e) {
  return new Proxy({ props: e }, Ja);
}
function I(e, t, r, n) {
  var i = !oa || (r & la) !== 0, a = (r & sa) !== 0, o = (r & fa) !== 0, s = (
    /** @type {V} */
    n
  ), u = !0, l = (
    /** @type {Derived<V> | undefined} */
    void 0
  ), c = () => o && i ? (l ??= xr(
    /** @type {() => V} */
    n
  ), f(l)) : (u && (u = !1, s = o ? ae(
    /** @type {() => V} */
    n
  ) : (
    /** @type {V} */
    n
  )), s);
  let d;
  if (a) {
    var v = Vr in e || jn in e;
    d = Er(e, t)?.set ?? (v && t in e ? (b) => e[t] = b : void 0);
  }
  var w, g = !1;
  a ? [w, g] = Ea(() => (
    /** @type {V} */
    e[t]
  )) : w = /** @type {V} */
  e[t], w === void 0 && n !== void 0 && (w = c(), d && (i && ia(), d(w)));
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
  }, i && (r & aa) === 0)
    return x;
  if (d) {
    var H = e.$$legacy;
    return (
      /** @type {() => V} */
      (function(b, y) {
        return arguments.length > 0 ? ((!i || !y || H || g) && d(y ? x() : b), b) : x();
      })
    );
  }
  var h = !1, m = ((r & ua) !== 0 ? xr : Dn)(() => (h = !1, x()));
  a && f(m);
  var p = (
    /** @type {Effect} */
    qt
  );
  return (
    /** @type {() => V} */
    (function(b, y) {
      if (arguments.length > 0) {
        const A = y ? f(m) : i && a ? wt(b) : b;
        return E(m, A), h = !0, s !== void 0 && (s = A), b;
      }
      return ca && h || (p.f & kn) !== 0 ? m.v : f(m);
    })
  );
}
ha();
var Ka = /* @__PURE__ */ zn('<svg class="resize-handle svelte-1stq1b1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><line x1="1" y1="9" x2="9" y2="1" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line><line x1="5" y1="9" x2="9" y2="5" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line></svg>'), dn = /* @__PURE__ */ se("<!> <!>", 1), $a = /* @__PURE__ */ se('<div class="placeholder svelte-1stq1b1"></div>');
function es(e, t) {
  Jt(t, !1);
  let r = I(t, "height", 8, void 0), n = I(t, "min_height", 8, void 0), i = I(t, "max_height", 8, void 0), a = I(t, "width", 8, void 0), o = I(t, "elem_id", 8, ""), s = I(t, "elem_classes", 24, () => []), u = I(t, "variant", 8, "solid"), l = I(t, "border_mode", 8, "base"), c = I(t, "padding", 8, !0), d = I(t, "type", 8, "normal"), v = I(t, "test_id", 8, void 0), w = I(t, "explicit_call", 8, !1), g = I(t, "container", 8, !0), x = I(t, "visible", 8, !0), H = I(t, "allow_overflow", 8, !0), h = I(t, "overflow_behavior", 8, "auto"), m = I(t, "scale", 8, null), p = I(t, "min_width", 8, 0), b = I(t, "flex", 12, !1), y = I(t, "resizable", 8, !1), A = I(t, "rtl", 8, !1), P = I(t, "fullscreen", 12, !1), B = I(t, "label", 8, void 0), D = Ye(P()), L = Ye(), Z = d() === "fieldset" ? "fieldset" : "div", ne = Ye(0), K = Ye(0), G = Ye(null);
  function ke(J) {
    P() && J.key === "Escape" && P(!1);
  }
  const He = (J) => {
    if (J !== void 0) {
      if (typeof J == "number")
        return J + "px";
      if (typeof J == "string")
        return J;
    }
  }, Ue = (J) => {
    let de = J.clientY;
    const ve = (re) => {
      const ge = re.clientY - de;
      de = re.clientY, pa(L, f(L).style.height = `${f(L).offsetHeight + ge}px`);
    }, me = () => {
      window.removeEventListener("mousemove", ve), window.removeEventListener("mouseup", me);
    };
    window.addEventListener("mousemove", ve), window.addEventListener("mouseup", me);
  };
  rn(
    () => (xe(P()), f(D), f(L)),
    () => {
      P() !== f(D) && (E(D, P()), P() ? (E(G, f(L).getBoundingClientRect()), E(ne, f(L).offsetHeight), E(K, f(L).offsetWidth), window.addEventListener("keydown", ke)) : (E(G, null), window.removeEventListener("keydown", ke)));
    }
  ), rn(() => xe(x()), () => {
    x() || b(!1);
  }), da(), Wa();
  var pe = it(), Pe = fe(pe);
  {
    var dt = (J) => {
      var de = dn(), ve = fe(de);
      Ma(ve, () => Z, !1, (ge, Fe) => {
        Xr(ge, (be) => E(L, be), () => f(L)), qa(
          ge,
          (be, Le) => ({
            "data-testid": v(),
            id: o(),
            class: `block ${be ?? ""}`,
            dir: A() ? "rtl" : "ltr",
            "aria-label": B(),
            style: "",
            [Et]: {
              hidden: x() === "hidden",
              padded: c(),
              flex: b(),
              border_focus: l() === "focus",
              border_contrast: l() === "contrast",
              "hide-container": !w() && !g(),
              fullscreen: P(),
              animating: P() && f(G) !== null,
              "auto-margin": m() === null
            },
            [nt]: Le
          }),
          [
            () => (xe(s()), ae(() => s()?.join(" ") || "")),
            () => ({
              height: (xe(P()), xe(r()), ae(() => P() ? void 0 : He(r()))),
              "min-height": (xe(P()), xe(n()), ae(() => P() ? void 0 : He(n()))),
              "max-height": (xe(P()), xe(i()), ae(() => P() ? void 0 : He(i()))),
              "--start-top": (f(G), ae(() => f(G) ? `${f(G).top}px` : "0px")),
              "--start-left": (f(G), ae(() => f(G) ? `${f(G).left}px` : "0px")),
              "--start-width": (f(G), ae(() => f(G) ? `${f(G).width}px` : "0px")),
              "--start-height": (f(G), ae(() => f(G) ? `${f(G).height}px` : "0px")),
              width: (xe(P()), xe(a()), ae(() => P() ? void 0 : typeof a() == "number" ? `calc(min(${a()}px, 100%))` : He(a()))),
              "border-style": u(),
              overflow: H() ? h() : "hidden",
              "flex-grow": m(),
              "min-width": `calc(min(${p()}px, 100%))`
            })
          ],
          void 0,
          void 0,
          "svelte-1stq1b1"
        );
        var he = dn(), Ge = fe(he);
        Sr(Ge, t, "default", {});
        var Ee = j(Ge, 2);
        {
          var qe = (be) => {
            var Le = Ka();
            Oe("mousedown", Le, Ue), C(be, Le);
          };
          Y(Ee, (be) => {
            y() && be(qe);
          });
        }
        C(Fe, he);
      });
      var me = j(ve, 2);
      {
        var re = (ge) => {
          var Fe = $a();
          let he;
          Q(() => he = Ae(Fe, "", he, {
            height: f(ne) + "px",
            width: f(K) + "px"
          })), C(ge, Fe);
        };
        Y(me, (ge) => {
          P() && ge(re);
        });
      }
      C(J, de);
    };
    Y(Pe, (J) => {
      (x() === !0 || x() === "hidden") && J(dt);
    });
  }
  C(e, pe), Yt();
}
var ts = /* @__PURE__ */ se('<span class="svelte-vvirtv"> </span>'), rs = /* @__PURE__ */ se("<button><!> <div><!> <!></div></button>");
function pn(e, t) {
  let r = I(t, "label", 3, ""), n = I(t, "show_label", 3, !1), i = I(t, "pending", 3, !1), a = I(t, "size", 3, "small"), o = I(t, "padded", 3, !0), s = I(t, "highlight", 3, !1), u = I(t, "disabled", 3, !1), l = I(t, "hasPopup", 3, !1), c = I(t, "color", 3, "var(--block-label-text-color)"), d = I(t, "transparent", 3, !1), v = I(t, "background", 3, "var(--block-background-fill)"), w = I(t, "border", 3, "transparent"), g = we(() => s() ? "var(--color-accent)" : c());
  var x = rs();
  let H, h;
  var m = te(x);
  {
    var p = (D) => {
      var L = ts(), Z = te(L);
      Q(() => ue(Z, r())), C(D, L);
    };
    Y(m, (D) => {
      n() && D(p);
    });
  }
  var b = j(m, 2);
  let y;
  var A = te(b);
  Oa(A, () => t.Icon, (D, L) => {
    L(D, {});
  });
  var P = j(A, 2);
  {
    var B = (D) => {
      var L = it(), Z = fe(L);
      Ha(Z, () => t.children), C(D, L);
    };
    Y(P, (D) => {
      t.children && D(B);
    });
  }
  Q(() => {
    H = De(x, 1, "icon-button svelte-vvirtv", null, H, {
      pending: i(),
      padded: o(),
      highlight: s(),
      transparent: d()
    }), x.disabled = u(), at(x, "aria-label", r()), at(x, "aria-haspopup", l()), at(x, "title", r()), h = Ae(x, "", h, {
      "--border-color": w(),
      color: !u() && f(g) ? f(g) : "var(--block-label-text-color)",
      "--bg-color": u() ? "auto" : v()
    }), y = De(b, 1, "svelte-vvirtv", null, y, {
      "x-small": a() === "x-small",
      small: a() === "small",
      large: a() === "large",
      medium: a() === "medium"
    });
  }), Fn("click", x, function(...D) {
    t.onclick?.apply(this, D);
  }), C(e, x);
}
Zt(["click"]);
var ns = /* @__PURE__ */ zn('<svg width="100%" height="100%" viewBox="0 0 24 24" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:space="preserve" stroke="currentColor" style="fill-rule:evenodd;clip-rule:evenodd;stroke-linecap:round;stroke-linejoin:round;"><g transform="matrix(1.14096,-0.140958,-0.140958,1.14096,-0.0559523,0.0559523)"><path d="M18,6L6.087,17.913" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></g><path d="M4.364,4.364L19.636,19.636" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></svg>');
function vn(e) {
  var t = ns();
  C(e, t);
}
const is = [
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
], mn = {
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
is.reduce((e, { color: t, primary: r, secondary: n }) => ({
  ...e,
  [t]: {
    primary: mn[t][r],
    secondary: mn[t][n]
  }
}), {});
function as(e) {
  return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, "default") ? e.default : e;
}
var cr, gn;
function ss() {
  if (gn) return cr;
  gn = 1;
  var e = function(m) {
    return t(m) && !r(m);
  };
  function t(h) {
    return !!h && typeof h == "object";
  }
  function r(h) {
    var m = Object.prototype.toString.call(h);
    return m === "[object RegExp]" || m === "[object Date]" || a(h);
  }
  var n = typeof Symbol == "function" && Symbol.for, i = n ? /* @__PURE__ */ Symbol.for("react.element") : 60103;
  function a(h) {
    return h.$$typeof === i;
  }
  function o(h) {
    return Array.isArray(h) ? [] : {};
  }
  function s(h, m) {
    return m.clone !== !1 && m.isMergeableObject(h) ? x(o(h), h, m) : h;
  }
  function u(h, m, p) {
    return h.concat(m).map(function(b) {
      return s(b, p);
    });
  }
  function l(h, m) {
    if (!m.customMerge)
      return x;
    var p = m.customMerge(h);
    return typeof p == "function" ? p : x;
  }
  function c(h) {
    return Object.getOwnPropertySymbols ? Object.getOwnPropertySymbols(h).filter(function(m) {
      return Object.propertyIsEnumerable.call(h, m);
    }) : [];
  }
  function d(h) {
    return Object.keys(h).concat(c(h));
  }
  function v(h, m) {
    try {
      return m in h;
    } catch {
      return !1;
    }
  }
  function w(h, m) {
    return v(h, m) && !(Object.hasOwnProperty.call(h, m) && Object.propertyIsEnumerable.call(h, m));
  }
  function g(h, m, p) {
    var b = {};
    return p.isMergeableObject(h) && d(h).forEach(function(y) {
      b[y] = s(h[y], p);
    }), d(m).forEach(function(y) {
      w(h, y) || (v(h, y) && p.isMergeableObject(m[y]) ? b[y] = l(y, p)(h[y], m[y], p) : b[y] = s(m[y], p));
    }), b;
  }
  function x(h, m, p) {
    p = p || {}, p.arrayMerge = p.arrayMerge || u, p.isMergeableObject = p.isMergeableObject || e, p.cloneUnlessOtherwiseSpecified = s;
    var b = Array.isArray(m), y = Array.isArray(h), A = b === y;
    return A ? b ? p.arrayMerge(h, m, p) : g(h, m, p) : s(m, p);
  }
  x.all = function(m, p) {
    if (!Array.isArray(m))
      throw new Error("first argument should be an array");
    return m.reduce(function(b, y) {
      return x(b, y, p);
    }, {});
  };
  var H = x;
  return cr = H, cr;
}
var os = ss();
const ls = /* @__PURE__ */ as(os);
var Hr = function(e, t) {
  return Hr = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(r, n) {
    r.__proto__ = n;
  } || function(r, n) {
    for (var i in n) Object.prototype.hasOwnProperty.call(n, i) && (r[i] = n[i]);
  }, Hr(e, t);
};
function Kt(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Class extends value " + String(t) + " is not a constructor or null");
  Hr(e, t);
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
function us(e, t) {
  var r = {};
  for (var n in e) Object.prototype.hasOwnProperty.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && typeof Object.getOwnPropertySymbols == "function")
    for (var i = 0, n = Object.getOwnPropertySymbols(e); i < n.length; i++)
      t.indexOf(n[i]) < 0 && Object.prototype.propertyIsEnumerable.call(e, n[i]) && (r[n[i]] = e[n[i]]);
  return r;
}
function hr(e, t, r) {
  if (r || arguments.length === 2) for (var n = 0, i = t.length, a; n < i; n++)
    (a || !(n in t)) && (a || (a = Array.prototype.slice.call(t, 0, n)), a[n] = t[n]);
  return e.concat(a || Array.prototype.slice.call(t));
}
function dr(e, t) {
  var r = t && t.cache ? t.cache : ms, n = t && t.serializer ? t.serializer : ps, i = t && t.strategy ? t.strategy : hs;
  return i(e, {
    cache: r,
    serializer: n
  });
}
function fs(e) {
  return e == null || typeof e == "number" || typeof e == "boolean";
}
function cs(e, t, r, n) {
  var i = fs(n) ? n : r(n), a = t.get(i);
  return typeof a > "u" && (a = e.call(this, n), t.set(i, a)), a;
}
function Qn(e, t, r) {
  var n = Array.prototype.slice.call(arguments, 3), i = r(n), a = t.get(i);
  return typeof a > "u" && (a = e.apply(this, n), t.set(i, a)), a;
}
function Kn(e, t, r, n, i) {
  return r.bind(t, e, n, i);
}
function hs(e, t) {
  var r = e.length === 1 ? cs : Qn;
  return Kn(e, this, r, t.cache.create(), t.serializer);
}
function ds(e, t) {
  return Kn(e, this, Qn, t.cache.create(), t.serializer);
}
var ps = function() {
  return JSON.stringify(arguments);
}, vs = (
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
), ms = {
  create: function() {
    return new vs();
  }
}, pr = {
  variadic: ds
}, M;
(function(e) {
  e[e.EXPECT_ARGUMENT_CLOSING_BRACE = 1] = "EXPECT_ARGUMENT_CLOSING_BRACE", e[e.EMPTY_ARGUMENT = 2] = "EMPTY_ARGUMENT", e[e.MALFORMED_ARGUMENT = 3] = "MALFORMED_ARGUMENT", e[e.EXPECT_ARGUMENT_TYPE = 4] = "EXPECT_ARGUMENT_TYPE", e[e.INVALID_ARGUMENT_TYPE = 5] = "INVALID_ARGUMENT_TYPE", e[e.EXPECT_ARGUMENT_STYLE = 6] = "EXPECT_ARGUMENT_STYLE", e[e.INVALID_NUMBER_SKELETON = 7] = "INVALID_NUMBER_SKELETON", e[e.INVALID_DATE_TIME_SKELETON = 8] = "INVALID_DATE_TIME_SKELETON", e[e.EXPECT_NUMBER_SKELETON = 9] = "EXPECT_NUMBER_SKELETON", e[e.EXPECT_DATE_TIME_SKELETON = 10] = "EXPECT_DATE_TIME_SKELETON", e[e.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE = 11] = "UNCLOSED_QUOTE_IN_ARGUMENT_STYLE", e[e.EXPECT_SELECT_ARGUMENT_OPTIONS = 12] = "EXPECT_SELECT_ARGUMENT_OPTIONS", e[e.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE = 13] = "EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE = 14] = "INVALID_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR = 15] = "EXPECT_SELECT_ARGUMENT_SELECTOR", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR = 16] = "EXPECT_PLURAL_ARGUMENT_SELECTOR", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT = 17] = "EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT = 18] = "EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT", e[e.INVALID_PLURAL_ARGUMENT_SELECTOR = 19] = "INVALID_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_PLURAL_ARGUMENT_SELECTOR = 20] = "DUPLICATE_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_SELECT_ARGUMENT_SELECTOR = 21] = "DUPLICATE_SELECT_ARGUMENT_SELECTOR", e[e.MISSING_OTHER_CLAUSE = 22] = "MISSING_OTHER_CLAUSE", e[e.INVALID_TAG = 23] = "INVALID_TAG", e[e.INVALID_TAG_NAME = 25] = "INVALID_TAG_NAME", e[e.UNMATCHED_CLOSING_TAG = 26] = "UNMATCHED_CLOSING_TAG", e[e.UNCLOSED_TAG = 27] = "UNCLOSED_TAG";
})(M || (M = {}));
var X;
(function(e) {
  e[e.literal = 0] = "literal", e[e.argument = 1] = "argument", e[e.number = 2] = "number", e[e.date = 3] = "date", e[e.time = 4] = "time", e[e.select = 5] = "select", e[e.plural = 6] = "plural", e[e.pound = 7] = "pound", e[e.tag = 8] = "tag";
})(X || (X = {}));
var ot;
(function(e) {
  e[e.number = 0] = "number", e[e.dateTime = 1] = "dateTime";
})(ot || (ot = {}));
function bn(e) {
  return e.type === X.literal;
}
function gs(e) {
  return e.type === X.argument;
}
function $n(e) {
  return e.type === X.number;
}
function ei(e) {
  return e.type === X.date;
}
function ti(e) {
  return e.type === X.time;
}
function ri(e) {
  return e.type === X.select;
}
function ni(e) {
  return e.type === X.plural;
}
function bs(e) {
  return e.type === X.pound;
}
function ii(e) {
  return e.type === X.tag;
}
function ai(e) {
  return !!(e && typeof e == "object" && e.type === ot.number);
}
function Pr(e) {
  return !!(e && typeof e == "object" && e.type === ot.dateTime);
}
var si = /[ \xA0\u1680\u2000-\u200A\u202F\u205F\u3000]/, _s = /(?:[Eec]{1,6}|G{1,5}|[Qq]{1,5}|(?:[yYur]+|U{1,5})|[ML]{1,5}|d{1,2}|D{1,3}|F{1}|[abB]{1,5}|[hkHK]{1,2}|w{1,2}|W{1}|m{1,2}|s{1,2}|[zZOvVxX]{1,4})(?=([^']*'[^']*')*[^']*$)/g;
function ys(e) {
  var t = {};
  return e.replace(_s, function(r) {
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
var xs = /[\t-\r \x85\u200E\u200F\u2028\u2029]/i;
function Es(e) {
  if (e.length === 0)
    throw new Error("Number skeleton cannot be empty");
  for (var t = e.split(xs).filter(function(v) {
    return v.length > 0;
  }), r = [], n = 0, i = t; n < i.length; n++) {
    var a = i[n], o = a.split("/");
    if (o.length === 0)
      throw new Error("Invalid number skeleton");
    for (var s = o[0], u = o.slice(1), l = 0, c = u; l < c.length; l++) {
      var d = c[l];
      if (d.length === 0)
        throw new Error("Invalid number skeleton");
    }
    r.push({ stem: s, options: u });
  }
  return r;
}
function ws(e) {
  return e.replace(/^(.*?)-/, "");
}
var _n = /^\.(?:(0+)(\*)?|(#+)|(0+)(#+))$/g, oi = /^(@+)?(\+|#+)?[rs]?$/g, Ts = /(\*)(0+)|(#+)(0+)|(0+)/g, li = /^(0+)$/;
function yn(e) {
  var t = {};
  return e[e.length - 1] === "r" ? t.roundingPriority = "morePrecision" : e[e.length - 1] === "s" && (t.roundingPriority = "lessPrecision"), e.replace(oi, function(r, n, i) {
    return typeof i != "string" ? (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length) : i === "+" ? t.minimumSignificantDigits = n.length : n[0] === "#" ? t.maximumSignificantDigits = n.length : (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length + (typeof i == "string" ? i.length : 0)), "";
  }), t;
}
function ui(e) {
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
function Ss(e) {
  var t;
  if (e[0] === "E" && e[1] === "E" ? (t = {
    notation: "engineering"
  }, e = e.slice(2)) : e[0] === "E" && (t = {
    notation: "scientific"
  }, e = e.slice(1)), t) {
    var r = e.slice(0, 2);
    if (r === "+!" ? (t.signDisplay = "always", e = e.slice(2)) : r === "+?" && (t.signDisplay = "exceptZero", e = e.slice(2)), !li.test(e))
      throw new Error("Malformed concise eng/scientific notation");
    t.minimumIntegerDigits = e.length;
  }
  return t;
}
function xn(e) {
  var t = {}, r = ui(e);
  return r || t;
}
function As(e) {
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
        t.style = "unit", t.unit = ws(i.options[0]);
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
          return k(k({}, u), xn(l));
        }, {}));
        continue;
      case "engineering":
        t = k(k(k({}, t), { notation: "engineering" }), i.options.reduce(function(u, l) {
          return k(k({}, u), xn(l));
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
        i.options[0].replace(Ts, function(u, l, c, d, v, w) {
          if (l)
            t.minimumIntegerDigits = c.length;
          else {
            if (d && v)
              throw new Error("We currently do not support maximum integer digits");
            if (w)
              throw new Error("We currently do not support exact integer digits");
          }
          return "";
        });
        continue;
    }
    if (li.test(i.stem)) {
      t.minimumIntegerDigits = i.stem.length;
      continue;
    }
    if (_n.test(i.stem)) {
      if (i.options.length > 1)
        throw new RangeError("Fraction-precision stems only accept a single optional option");
      i.stem.replace(_n, function(u, l, c, d, v, w) {
        return c === "*" ? t.minimumFractionDigits = l.length : d && d[0] === "#" ? t.maximumFractionDigits = d.length : v && w ? (t.minimumFractionDigits = v.length, t.maximumFractionDigits = v.length + w.length) : (t.minimumFractionDigits = l.length, t.maximumFractionDigits = l.length), "";
      });
      var a = i.options[0];
      a === "w" ? t = k(k({}, t), { trailingZeroDisplay: "stripIfInteger" }) : a && (t = k(k({}, t), yn(a)));
      continue;
    }
    if (oi.test(i.stem)) {
      t = k(k({}, t), yn(i.stem));
      continue;
    }
    var o = ui(i.stem);
    o && (t = k(k({}, t), o));
    var s = Ss(i.stem);
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
function Hs(e, t) {
  for (var r = "", n = 0; n < e.length; n++) {
    var i = e.charAt(n);
    if (i === "j") {
      for (var a = 0; n + 1 < e.length && e.charAt(n + 1) === i; )
        a++, n++;
      var o = 1 + (a & 1), s = a < 2 ? 1 : 3 + (a >> 1), u = "a", l = Ps(t);
      for ((l == "H" || l == "k") && (s = 0); s-- > 0; )
        r += u;
      for (; o-- > 0; )
        r = l + r;
    } else i === "J" ? r += "H" : r += i;
  }
  return r;
}
function Ps(e) {
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
var vr, Is = new RegExp("^".concat(si.source, "*")), Bs = new RegExp("".concat(si.source, "*$"));
function R(e, t) {
  return { start: e, end: t };
}
var Os = !!String.prototype.startsWith && "_a".startsWith("a", 1), Ls = !!String.fromCodePoint, Ns = !!Object.fromEntries, Ms = !!String.prototype.codePointAt, Cs = !!String.prototype.trimStart, Rs = !!String.prototype.trimEnd, Ds = !!Number.isSafeInteger, ks = Ds ? Number.isSafeInteger : function(e) {
  return typeof e == "number" && isFinite(e) && Math.floor(e) === e && Math.abs(e) <= 9007199254740991;
}, Ir = !0;
try {
  var Us = ci("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  Ir = ((vr = Us.exec("a")) === null || vr === void 0 ? void 0 : vr[0]) === "a";
} catch {
  Ir = !1;
}
var En = Os ? (
  // Native
  function(t, r, n) {
    return t.startsWith(r, n);
  }
) : (
  // For IE11
  function(t, r, n) {
    return t.slice(n, n + r.length) === r;
  }
), Br = Ls ? String.fromCodePoint : (
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
), wn = (
  // native
  Ns ? Object.fromEntries : (
    // Ponyfill
    function(t) {
      for (var r = {}, n = 0, i = t; n < i.length; n++) {
        var a = i[n], o = a[0], s = a[1];
        r[o] = s;
      }
      return r;
    }
  )
), fi = Ms ? (
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
), Fs = Cs ? (
  // Native
  function(t) {
    return t.trimStart();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(Is, "");
  }
), Gs = Rs ? (
  // Native
  function(t) {
    return t.trimEnd();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(Bs, "");
  }
);
function ci(e, t) {
  return new RegExp(e, t);
}
var Or;
if (Ir) {
  var Tn = ci("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  Or = function(t, r) {
    var n;
    Tn.lastIndex = r;
    var i = Tn.exec(t);
    return (n = i[1]) !== null && n !== void 0 ? n : "";
  };
} else
  Or = function(t, r) {
    for (var n = []; ; ) {
      var i = fi(t, r);
      if (i === void 0 || hi(i) || Xs(i))
        break;
      n.push(i), r += i >= 65536 ? 2 : 1;
    }
    return Br.apply(void 0, n);
  };
var js = (
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
              location: R(s, this.clonePosition())
            });
          } else if (a === 60 && !this.ignoreTag && this.peek() === 47) {
            if (n)
              break;
            return this.error(M.UNMATCHED_CLOSING_TAG, R(this.clonePosition(), this.clonePosition()));
          } else if (a === 60 && !this.ignoreTag && Lr(this.peek() || 0)) {
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
          if (this.isEOF() || !Lr(this.char()))
            return this.error(M.INVALID_TAG, R(s, this.clonePosition()));
          var u = this.clonePosition(), l = this.parseTagName();
          return i !== l ? this.error(M.UNMATCHED_CLOSING_TAG, R(u, this.clonePosition())) : (this.bumpSpace(), this.bumpIf(">") ? {
            val: {
              type: X.tag,
              value: i,
              children: o,
              location: R(n, this.clonePosition())
            },
            err: null
          } : this.error(M.INVALID_TAG, R(s, this.clonePosition())));
        } else
          return this.error(M.UNCLOSED_TAG, R(n, this.clonePosition()));
      } else
        return this.error(M.INVALID_TAG, R(n, this.clonePosition()));
    }, e.prototype.parseTagName = function() {
      var t = this.offset();
      for (this.bump(); !this.isEOF() && zs(this.char()); )
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
        val: { type: X.literal, value: i, location: u },
        err: null
      };
    }, e.prototype.tryParseLeftAngleBracket = function() {
      return !this.isEOF() && this.char() === 60 && (this.ignoreTag || // If at the opening tag or closing tag position, bail.
      !Vs(this.peek() || 0)) ? (this.bump(), "<") : null;
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
      return Br.apply(void 0, r);
    }, e.prototype.tryParseUnquoted = function(t, r) {
      if (this.isEOF())
        return null;
      var n = this.char();
      return n === 60 || n === 123 || n === 35 && (r === "plural" || r === "selectordinal") || n === 125 && t > 0 ? null : (this.bump(), Br(n));
    }, e.prototype.parseArgument = function(t, r) {
      var n = this.clonePosition();
      if (this.bump(), this.bumpSpace(), this.isEOF())
        return this.error(M.EXPECT_ARGUMENT_CLOSING_BRACE, R(n, this.clonePosition()));
      if (this.char() === 125)
        return this.bump(), this.error(M.EMPTY_ARGUMENT, R(n, this.clonePosition()));
      var i = this.parseIdentifierIfPossible().value;
      if (!i)
        return this.error(M.MALFORMED_ARGUMENT, R(n, this.clonePosition()));
      if (this.bumpSpace(), this.isEOF())
        return this.error(M.EXPECT_ARGUMENT_CLOSING_BRACE, R(n, this.clonePosition()));
      switch (this.char()) {
        // Simple argument: `{name}`
        case 125:
          return this.bump(), {
            val: {
              type: X.argument,
              // value does not include the opening and closing braces.
              value: i,
              location: R(n, this.clonePosition())
            },
            err: null
          };
        // Argument with options: `{name, format, ...}`
        case 44:
          return this.bump(), this.bumpSpace(), this.isEOF() ? this.error(M.EXPECT_ARGUMENT_CLOSING_BRACE, R(n, this.clonePosition())) : this.parseArgumentOptions(t, r, i, n);
        default:
          return this.error(M.MALFORMED_ARGUMENT, R(n, this.clonePosition()));
      }
    }, e.prototype.parseIdentifierIfPossible = function() {
      var t = this.clonePosition(), r = this.offset(), n = Or(this.message, r), i = r + n.length;
      this.bumpTo(i);
      var a = this.clonePosition(), o = R(t, a);
      return { value: n, location: o };
    }, e.prototype.parseArgumentOptions = function(t, r, n, i) {
      var a, o = this.clonePosition(), s = this.parseIdentifierIfPossible().value, u = this.clonePosition();
      switch (s) {
        case "":
          return this.error(M.EXPECT_ARGUMENT_TYPE, R(o, u));
        case "number":
        case "date":
        case "time": {
          this.bumpSpace();
          var l = null;
          if (this.bumpIf(",")) {
            this.bumpSpace();
            var c = this.clonePosition(), d = this.parseSimpleArgStyleIfPossible();
            if (d.err)
              return d;
            var v = Gs(d.val);
            if (v.length === 0)
              return this.error(M.EXPECT_ARGUMENT_STYLE, R(this.clonePosition(), this.clonePosition()));
            var w = R(c, this.clonePosition());
            l = { style: v, styleLocation: w };
          }
          var g = this.tryParseArgumentClose(i);
          if (g.err)
            return g;
          var x = R(i, this.clonePosition());
          if (l && En(l?.style, "::", 0)) {
            var H = Fs(l.style.slice(2));
            if (s === "number") {
              var d = this.parseNumberSkeletonFromString(H, l.styleLocation);
              return d.err ? d : {
                val: { type: X.number, value: n, location: x, style: d.val },
                err: null
              };
            } else {
              if (H.length === 0)
                return this.error(M.EXPECT_DATE_TIME_SKELETON, x);
              var h = H;
              this.locale && (h = Hs(H, this.locale));
              var v = {
                type: ot.dateTime,
                pattern: h,
                location: l.styleLocation,
                parsedOptions: this.shouldParseSkeletons ? ys(h) : {}
              }, m = s === "date" ? X.date : X.time;
              return {
                val: { type: m, value: n, location: x, style: v },
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
          var p = this.clonePosition();
          if (this.bumpSpace(), !this.bumpIf(","))
            return this.error(M.EXPECT_SELECT_ARGUMENT_OPTIONS, R(p, k({}, p)));
          this.bumpSpace();
          var b = this.parseIdentifierIfPossible(), y = 0;
          if (s !== "select" && b.value === "offset") {
            if (!this.bumpIf(":"))
              return this.error(M.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, R(this.clonePosition(), this.clonePosition()));
            this.bumpSpace();
            var d = this.tryParseDecimalInteger(M.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, M.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE);
            if (d.err)
              return d;
            this.bumpSpace(), b = this.parseIdentifierIfPossible(), y = d.val;
          }
          var A = this.tryParsePluralOrSelectOptions(t, s, r, b);
          if (A.err)
            return A;
          var g = this.tryParseArgumentClose(i);
          if (g.err)
            return g;
          var P = R(i, this.clonePosition());
          return s === "select" ? {
            val: {
              type: X.select,
              value: n,
              options: wn(A.val),
              location: P
            },
            err: null
          } : {
            val: {
              type: X.plural,
              value: n,
              options: wn(A.val),
              offset: y,
              pluralType: s === "plural" ? "cardinal" : "ordinal",
              location: P
            },
            err: null
          };
        }
        default:
          return this.error(M.INVALID_ARGUMENT_TYPE, R(o, u));
      }
    }, e.prototype.tryParseArgumentClose = function(t) {
      return this.isEOF() || this.char() !== 125 ? this.error(M.EXPECT_ARGUMENT_CLOSING_BRACE, R(t, this.clonePosition())) : (this.bump(), { val: !0, err: null });
    }, e.prototype.parseSimpleArgStyleIfPossible = function() {
      for (var t = 0, r = this.clonePosition(); !this.isEOF(); ) {
        var n = this.char();
        switch (n) {
          case 39: {
            this.bump();
            var i = this.clonePosition();
            if (!this.bumpUntil("'"))
              return this.error(M.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE, R(i, this.clonePosition()));
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
        n = Es(t);
      } catch {
        return this.error(M.INVALID_NUMBER_SKELETON, r);
      }
      return {
        val: {
          type: ot.number,
          tokens: n,
          location: r,
          parsedOptions: this.shouldParseSkeletons ? As(n) : {}
        },
        err: null
      };
    }, e.prototype.tryParsePluralOrSelectOptions = function(t, r, n, i) {
      for (var a, o = !1, s = [], u = /* @__PURE__ */ new Set(), l = i.value, c = i.location; ; ) {
        if (l.length === 0) {
          var d = this.clonePosition();
          if (r !== "select" && this.bumpIf("=")) {
            var v = this.tryParseDecimalInteger(M.EXPECT_PLURAL_ARGUMENT_SELECTOR, M.INVALID_PLURAL_ARGUMENT_SELECTOR);
            if (v.err)
              return v;
            c = R(d, this.clonePosition()), l = this.message.slice(d.offset, this.offset());
          } else
            break;
        }
        if (u.has(l))
          return this.error(r === "select" ? M.DUPLICATE_SELECT_ARGUMENT_SELECTOR : M.DUPLICATE_PLURAL_ARGUMENT_SELECTOR, c);
        l === "other" && (o = !0), this.bumpSpace();
        var w = this.clonePosition();
        if (!this.bumpIf("{"))
          return this.error(r === "select" ? M.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT : M.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT, R(this.clonePosition(), this.clonePosition()));
        var g = this.parseMessage(t + 1, r, n);
        if (g.err)
          return g;
        var x = this.tryParseArgumentClose(w);
        if (x.err)
          return x;
        s.push([
          l,
          {
            value: g.val,
            location: R(w, this.clonePosition())
          }
        ]), u.add(l), this.bumpSpace(), a = this.parseIdentifierIfPossible(), l = a.value, c = a.location;
      }
      return s.length === 0 ? this.error(r === "select" ? M.EXPECT_SELECT_ARGUMENT_SELECTOR : M.EXPECT_PLURAL_ARGUMENT_SELECTOR, R(this.clonePosition(), this.clonePosition())) : this.requiresOtherClause && !o ? this.error(M.MISSING_OTHER_CLAUSE, R(this.clonePosition(), this.clonePosition())) : { val: s, err: null };
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
      return a ? (o *= n, ks(o) ? { val: o, err: null } : this.error(r, u)) : this.error(t, u);
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
      var r = fi(this.message, t);
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
      if (En(this.message, t, this.offset())) {
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
      for (; !this.isEOF() && hi(this.char()); )
        this.bump();
    }, e.prototype.peek = function() {
      if (this.isEOF())
        return null;
      var t = this.char(), r = this.offset(), n = this.message.charCodeAt(r + (t >= 65536 ? 2 : 1));
      return n ?? null;
    }, e;
  })()
);
function Lr(e) {
  return e >= 97 && e <= 122 || e >= 65 && e <= 90;
}
function Vs(e) {
  return Lr(e) || e === 47;
}
function zs(e) {
  return e === 45 || e === 46 || e >= 48 && e <= 57 || e === 95 || e >= 97 && e <= 122 || e >= 65 && e <= 90 || e == 183 || e >= 192 && e <= 214 || e >= 216 && e <= 246 || e >= 248 && e <= 893 || e >= 895 && e <= 8191 || e >= 8204 && e <= 8205 || e >= 8255 && e <= 8256 || e >= 8304 && e <= 8591 || e >= 11264 && e <= 12271 || e >= 12289 && e <= 55295 || e >= 63744 && e <= 64975 || e >= 65008 && e <= 65533 || e >= 65536 && e <= 983039;
}
function hi(e) {
  return e >= 9 && e <= 13 || e === 32 || e === 133 || e >= 8206 && e <= 8207 || e === 8232 || e === 8233;
}
function Xs(e) {
  return e >= 33 && e <= 35 || e === 36 || e >= 37 && e <= 39 || e === 40 || e === 41 || e === 42 || e === 43 || e === 44 || e === 45 || e >= 46 && e <= 47 || e >= 58 && e <= 59 || e >= 60 && e <= 62 || e >= 63 && e <= 64 || e === 91 || e === 92 || e === 93 || e === 94 || e === 96 || e === 123 || e === 124 || e === 125 || e === 126 || e === 161 || e >= 162 && e <= 165 || e === 166 || e === 167 || e === 169 || e === 171 || e === 172 || e === 174 || e === 176 || e === 177 || e === 182 || e === 187 || e === 191 || e === 215 || e === 247 || e >= 8208 && e <= 8213 || e >= 8214 && e <= 8215 || e === 8216 || e === 8217 || e === 8218 || e >= 8219 && e <= 8220 || e === 8221 || e === 8222 || e === 8223 || e >= 8224 && e <= 8231 || e >= 8240 && e <= 8248 || e === 8249 || e === 8250 || e >= 8251 && e <= 8254 || e >= 8257 && e <= 8259 || e === 8260 || e === 8261 || e === 8262 || e >= 8263 && e <= 8273 || e === 8274 || e === 8275 || e >= 8277 && e <= 8286 || e >= 8592 && e <= 8596 || e >= 8597 && e <= 8601 || e >= 8602 && e <= 8603 || e >= 8604 && e <= 8607 || e === 8608 || e >= 8609 && e <= 8610 || e === 8611 || e >= 8612 && e <= 8613 || e === 8614 || e >= 8615 && e <= 8621 || e === 8622 || e >= 8623 && e <= 8653 || e >= 8654 && e <= 8655 || e >= 8656 && e <= 8657 || e === 8658 || e === 8659 || e === 8660 || e >= 8661 && e <= 8691 || e >= 8692 && e <= 8959 || e >= 8960 && e <= 8967 || e === 8968 || e === 8969 || e === 8970 || e === 8971 || e >= 8972 && e <= 8991 || e >= 8992 && e <= 8993 || e >= 8994 && e <= 9e3 || e === 9001 || e === 9002 || e >= 9003 && e <= 9083 || e === 9084 || e >= 9085 && e <= 9114 || e >= 9115 && e <= 9139 || e >= 9140 && e <= 9179 || e >= 9180 && e <= 9185 || e >= 9186 && e <= 9254 || e >= 9255 && e <= 9279 || e >= 9280 && e <= 9290 || e >= 9291 && e <= 9311 || e >= 9472 && e <= 9654 || e === 9655 || e >= 9656 && e <= 9664 || e === 9665 || e >= 9666 && e <= 9719 || e >= 9720 && e <= 9727 || e >= 9728 && e <= 9838 || e === 9839 || e >= 9840 && e <= 10087 || e === 10088 || e === 10089 || e === 10090 || e === 10091 || e === 10092 || e === 10093 || e === 10094 || e === 10095 || e === 10096 || e === 10097 || e === 10098 || e === 10099 || e === 10100 || e === 10101 || e >= 10132 && e <= 10175 || e >= 10176 && e <= 10180 || e === 10181 || e === 10182 || e >= 10183 && e <= 10213 || e === 10214 || e === 10215 || e === 10216 || e === 10217 || e === 10218 || e === 10219 || e === 10220 || e === 10221 || e === 10222 || e === 10223 || e >= 10224 && e <= 10239 || e >= 10240 && e <= 10495 || e >= 10496 && e <= 10626 || e === 10627 || e === 10628 || e === 10629 || e === 10630 || e === 10631 || e === 10632 || e === 10633 || e === 10634 || e === 10635 || e === 10636 || e === 10637 || e === 10638 || e === 10639 || e === 10640 || e === 10641 || e === 10642 || e === 10643 || e === 10644 || e === 10645 || e === 10646 || e === 10647 || e === 10648 || e >= 10649 && e <= 10711 || e === 10712 || e === 10713 || e === 10714 || e === 10715 || e >= 10716 && e <= 10747 || e === 10748 || e === 10749 || e >= 10750 && e <= 11007 || e >= 11008 && e <= 11055 || e >= 11056 && e <= 11076 || e >= 11077 && e <= 11078 || e >= 11079 && e <= 11084 || e >= 11085 && e <= 11123 || e >= 11124 && e <= 11125 || e >= 11126 && e <= 11157 || e === 11158 || e >= 11159 && e <= 11263 || e >= 11776 && e <= 11777 || e === 11778 || e === 11779 || e === 11780 || e === 11781 || e >= 11782 && e <= 11784 || e === 11785 || e === 11786 || e === 11787 || e === 11788 || e === 11789 || e >= 11790 && e <= 11798 || e === 11799 || e >= 11800 && e <= 11801 || e === 11802 || e === 11803 || e === 11804 || e === 11805 || e >= 11806 && e <= 11807 || e === 11808 || e === 11809 || e === 11810 || e === 11811 || e === 11812 || e === 11813 || e === 11814 || e === 11815 || e === 11816 || e === 11817 || e >= 11818 && e <= 11822 || e === 11823 || e >= 11824 && e <= 11833 || e >= 11834 && e <= 11835 || e >= 11836 && e <= 11839 || e === 11840 || e === 11841 || e === 11842 || e >= 11843 && e <= 11855 || e >= 11856 && e <= 11857 || e === 11858 || e >= 11859 && e <= 11903 || e >= 12289 && e <= 12291 || e === 12296 || e === 12297 || e === 12298 || e === 12299 || e === 12300 || e === 12301 || e === 12302 || e === 12303 || e === 12304 || e === 12305 || e >= 12306 && e <= 12307 || e === 12308 || e === 12309 || e === 12310 || e === 12311 || e === 12312 || e === 12313 || e === 12314 || e === 12315 || e === 12316 || e === 12317 || e >= 12318 && e <= 12319 || e === 12320 || e === 12336 || e === 64830 || e === 64831 || e >= 65093 && e <= 65094;
}
function Nr(e) {
  e.forEach(function(t) {
    if (delete t.location, ri(t) || ni(t))
      for (var r in t.options)
        delete t.options[r].location, Nr(t.options[r].value);
    else $n(t) && ai(t.style) || (ei(t) || ti(t)) && Pr(t.style) ? delete t.style.location : ii(t) && Nr(t.children);
  });
}
function qs(e, t) {
  t === void 0 && (t = {}), t = k({ shouldParseSkeletons: !0, requiresOtherClause: !0 }, t);
  var r = new js(e, t).parse();
  if (r.err) {
    var n = SyntaxError(M[r.err.kind]);
    throw n.location = r.err.location, n.originalMessage = r.err.message, n;
  }
  return t?.captureLocation || Nr(r.val), r.val;
}
var lt;
(function(e) {
  e.MISSING_VALUE = "MISSING_VALUE", e.INVALID_VALUE = "INVALID_VALUE", e.MISSING_INTL_API = "MISSING_INTL_API";
})(lt || (lt = {}));
var $t = (
  /** @class */
  (function(e) {
    Kt(t, e);
    function t(r, n, i) {
      var a = e.call(this, r) || this;
      return a.code = n, a.originalMessage = i, a;
    }
    return t.prototype.toString = function() {
      return "[formatjs Error: ".concat(this.code, "] ").concat(this.message);
    }, t;
  })(Error)
), Sn = (
  /** @class */
  (function(e) {
    Kt(t, e);
    function t(r, n, i, a) {
      return e.call(this, 'Invalid values for "'.concat(r, '": "').concat(n, '". Options are "').concat(Object.keys(i).join('", "'), '"'), lt.INVALID_VALUE, a) || this;
    }
    return t;
  })($t)
), Ws = (
  /** @class */
  (function(e) {
    Kt(t, e);
    function t(r, n, i) {
      return e.call(this, 'Value for "'.concat(r, '" must be of type ').concat(n), lt.INVALID_VALUE, i) || this;
    }
    return t;
  })($t)
), Zs = (
  /** @class */
  (function(e) {
    Kt(t, e);
    function t(r, n) {
      return e.call(this, 'The intl string context variable "'.concat(r, '" was not provided to the string "').concat(n, '"'), lt.MISSING_VALUE, n) || this;
    }
    return t;
  })($t)
), ce;
(function(e) {
  e[e.literal = 0] = "literal", e[e.object = 1] = "object";
})(ce || (ce = {}));
function Ys(e) {
  return e.length < 2 ? e : e.reduce(function(t, r) {
    var n = t[t.length - 1];
    return !n || n.type !== ce.literal || r.type !== ce.literal ? t.push(r) : n.value += r.value, t;
  }, []);
}
function Js(e) {
  return typeof e == "function";
}
function Ft(e, t, r, n, i, a, o) {
  if (e.length === 1 && bn(e[0]))
    return [
      {
        type: ce.literal,
        value: e[0].value
      }
    ];
  for (var s = [], u = 0, l = e; u < l.length; u++) {
    var c = l[u];
    if (bn(c)) {
      s.push({
        type: ce.literal,
        value: c.value
      });
      continue;
    }
    if (bs(c)) {
      typeof a == "number" && s.push({
        type: ce.literal,
        value: r.getNumberFormat(t).format(a)
      });
      continue;
    }
    var d = c.value;
    if (!(i && d in i))
      throw new Zs(d, o);
    var v = i[d];
    if (gs(c)) {
      (!v || typeof v == "string" || typeof v == "number") && (v = typeof v == "string" || typeof v == "number" ? String(v) : ""), s.push({
        type: typeof v == "string" ? ce.literal : ce.object,
        value: v
      });
      continue;
    }
    if (ei(c)) {
      var w = typeof c.style == "string" ? n.date[c.style] : Pr(c.style) ? c.style.parsedOptions : void 0;
      s.push({
        type: ce.literal,
        value: r.getDateTimeFormat(t, w).format(v)
      });
      continue;
    }
    if (ti(c)) {
      var w = typeof c.style == "string" ? n.time[c.style] : Pr(c.style) ? c.style.parsedOptions : n.time.medium;
      s.push({
        type: ce.literal,
        value: r.getDateTimeFormat(t, w).format(v)
      });
      continue;
    }
    if ($n(c)) {
      var w = typeof c.style == "string" ? n.number[c.style] : ai(c.style) ? c.style.parsedOptions : void 0;
      w && w.scale && (v = v * (w.scale || 1)), s.push({
        type: ce.literal,
        value: r.getNumberFormat(t, w).format(v)
      });
      continue;
    }
    if (ii(c)) {
      var g = c.children, x = c.value, H = i[x];
      if (!Js(H))
        throw new Ws(x, "function", o);
      var h = Ft(g, t, r, n, i, a), m = H(h.map(function(y) {
        return y.value;
      }));
      Array.isArray(m) || (m = [m]), s.push.apply(s, m.map(function(y) {
        return {
          type: typeof y == "string" ? ce.literal : ce.object,
          value: y
        };
      }));
    }
    if (ri(c)) {
      var p = c.options[v] || c.options.other;
      if (!p)
        throw new Sn(c.value, v, Object.keys(c.options), o);
      s.push.apply(s, Ft(p.value, t, r, n, i));
      continue;
    }
    if (ni(c)) {
      var p = c.options["=".concat(v)];
      if (!p) {
        if (!Intl.PluralRules)
          throw new $t(`Intl.PluralRules is not available in this environment.
Try polyfilling it using "@formatjs/intl-pluralrules"
`, lt.MISSING_INTL_API, o);
        var b = r.getPluralRules(t, { type: c.pluralType }).select(v - (c.offset || 0));
        p = c.options[b] || c.options.other;
      }
      if (!p)
        throw new Sn(c.value, v, Object.keys(c.options), o);
      s.push.apply(s, Ft(p.value, t, r, n, i, v - (c.offset || 0)));
      continue;
    }
  }
  return Ys(s);
}
function Qs(e, t) {
  return t ? k(k(k({}, e || {}), t || {}), Object.keys(e).reduce(function(r, n) {
    return r[n] = k(k({}, e[n]), t[n] || {}), r;
  }, {})) : e;
}
function Ks(e, t) {
  return t ? Object.keys(e).reduce(function(r, n) {
    return r[n] = Qs(e[n], t[n]), r;
  }, k({}, e)) : e;
}
function mr(e) {
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
function $s(e) {
  return e === void 0 && (e = {
    number: {},
    dateTime: {},
    pluralRules: {}
  }), {
    getNumberFormat: dr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.NumberFormat).bind.apply(t, hr([void 0], r, !1)))();
    }, {
      cache: mr(e.number),
      strategy: pr.variadic
    }),
    getDateTimeFormat: dr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.DateTimeFormat).bind.apply(t, hr([void 0], r, !1)))();
    }, {
      cache: mr(e.dateTime),
      strategy: pr.variadic
    }),
    getPluralRules: dr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.PluralRules).bind.apply(t, hr([void 0], r, !1)))();
    }, {
      cache: mr(e.pluralRules),
      strategy: pr.variadic
    })
  };
}
var eo = (
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
        var c = l.reduce(function(d, v) {
          return !d.length || v.type !== ce.literal || typeof d[d.length - 1] != "string" ? d.push(v.value) : d[d.length - 1] += v.value, d;
        }, []);
        return c.length <= 1 ? c[0] || "" : c;
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
        var s = us(o, ["formatters"]);
        this.ast = e.__parse(t, k(k({}, s), { locale: this.resolvedLocale }));
      } else
        this.ast = t;
      if (!Array.isArray(this.ast))
        throw new TypeError("A message must be provided as a String or AST.");
      this.formats = Ks(e.formats, n), this.formatters = i && i.formatters || $s(this.formatterCache);
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
    }, e.__parse = qs, e.formats = {
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
function to(e, t) {
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
const ze = {}, ro = (e, t, r) => r && (t in ze || (ze[t] = {}), e in ze[t] || (ze[t][e] = r), r), di = (e, t) => {
  if (t == null)
    return;
  if (t in ze && e in ze[t])
    return ze[t][e];
  const r = er(t);
  for (let n = 0; n < r.length; n++) {
    const i = r[n], a = io(i, e);
    if (a)
      return ro(e, t, a);
  }
};
let qr;
const Ht = At({});
function no(e) {
  return qr[e] || null;
}
function pi(e) {
  return e in qr;
}
function io(e, t) {
  if (!pi(e))
    return null;
  const r = no(e);
  return to(r, t);
}
function ao(e) {
  if (e == null)
    return;
  const t = er(e);
  for (let r = 0; r < t.length; r++) {
    const n = t[r];
    if (pi(n))
      return n;
  }
}
function so(e, ...t) {
  delete ze[e], Ht.update((r) => (r[e] = ls.all([r[e] || {}, ...t]), r));
}
ft(
  [Ht],
  ([e]) => Object.keys(e)
);
Ht.subscribe((e) => qr = e);
const Gt = {};
function oo(e, t) {
  Gt[e].delete(t), Gt[e].size === 0 && delete Gt[e];
}
function vi(e) {
  return Gt[e];
}
function lo(e) {
  return er(e).map((t) => {
    const r = vi(t);
    return [t, r ? [...r] : []];
  }).filter(([, t]) => t.length > 0);
}
function Mr(e) {
  return e == null ? !1 : er(e).some(
    (t) => {
      var r;
      return (r = vi(t)) == null ? void 0 : r.size;
    }
  );
}
function uo(e, t) {
  return Promise.all(
    t.map((n) => (oo(e, n), n().then((i) => i.default || i)))
  ).then((n) => so(e, ...n));
}
const yt = {};
function mi(e) {
  if (!Mr(e))
    return e in yt ? yt[e] : Promise.resolve();
  const t = lo(e);
  return yt[e] = Promise.all(
    t.map(
      ([r, n]) => uo(r, n)
    )
  ).then(() => {
    if (Mr(e))
      return mi(e);
    delete yt[e];
  }), yt[e];
}
const fo = {
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
}, co = {
  fallbackLocale: null,
  loadingDelay: 200,
  formats: fo,
  warnOnMissingMessages: !0,
  handleMissingMessage: void 0,
  ignoreTag: !0
}, ho = co;
function ut() {
  return ho;
}
const gr = At(!1);
var po = Object.defineProperty, vo = Object.defineProperties, mo = Object.getOwnPropertyDescriptors, An = Object.getOwnPropertySymbols, go = Object.prototype.hasOwnProperty, bo = Object.prototype.propertyIsEnumerable, Hn = (e, t, r) => t in e ? po(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, _o = (e, t) => {
  for (var r in t || (t = {}))
    go.call(t, r) && Hn(e, r, t[r]);
  if (An)
    for (var r of An(t))
      bo.call(t, r) && Hn(e, r, t[r]);
  return e;
}, yo = (e, t) => vo(e, mo(t));
let Cr;
const zt = At(null);
function Pn(e) {
  return e.split("-").map((t, r, n) => n.slice(0, r + 1).join("-")).reverse();
}
function er(e, t = ut().fallbackLocale) {
  const r = Pn(e);
  return t ? [.../* @__PURE__ */ new Set([...r, ...Pn(t)])] : r;
}
function $e() {
  return Cr ?? void 0;
}
zt.subscribe((e) => {
  Cr = e ?? void 0, typeof window < "u" && e != null && document.documentElement.setAttribute("lang", e);
});
const xo = (e) => {
  if (e && ao(e) && Mr(e)) {
    const { loadingDelay: t } = ut();
    let r;
    return typeof window < "u" && $e() != null && t ? r = window.setTimeout(
      () => gr.set(!0),
      t
    ) : gr.set(!0), mi(e).then(() => {
      zt.set(e);
    }).finally(() => {
      clearTimeout(r), gr.set(!1);
    });
  }
  return zt.set(e);
}, ct = yo(_o({}, zt), {
  set: xo
}), tr = (e) => {
  const t = /* @__PURE__ */ Object.create(null);
  return (n) => {
    const i = JSON.stringify(n);
    return i in t ? t[i] : t[i] = e(n);
  };
};
var Eo = Object.defineProperty, Xt = Object.getOwnPropertySymbols, gi = Object.prototype.hasOwnProperty, bi = Object.prototype.propertyIsEnumerable, In = (e, t, r) => t in e ? Eo(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, Wr = (e, t) => {
  for (var r in t || (t = {}))
    gi.call(t, r) && In(e, r, t[r]);
  if (Xt)
    for (var r of Xt(t))
      bi.call(t, r) && In(e, r, t[r]);
  return e;
}, ht = (e, t) => {
  var r = {};
  for (var n in e)
    gi.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && Xt)
    for (var n of Xt(e))
      t.indexOf(n) < 0 && bi.call(e, n) && (r[n] = e[n]);
  return r;
};
const Tt = (e, t) => {
  const { formats: r } = ut();
  if (e in r && t in r[e])
    return r[e][t];
  throw new Error(`[svelte-i18n] Unknown "${t}" ${e} format.`);
}, wo = tr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = ht(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format numbers');
    return n && (i = Tt("number", n)), new Intl.NumberFormat(r, i);
  }
), To = tr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = ht(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format dates');
    return n ? i = Tt("date", n) : Object.keys(i).length === 0 && (i = Tt("date", "short")), new Intl.DateTimeFormat(r, i);
  }
), So = tr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = ht(t, ["locale", "format"]);
    if (r == null)
      throw new Error(
        '[svelte-i18n] A "locale" must be set to format time values'
      );
    return n ? i = Tt("time", n) : Object.keys(i).length === 0 && (i = Tt("time", "short")), new Intl.DateTimeFormat(r, i);
  }
), Ao = (e = {}) => {
  var t = e, {
    locale: r = $e()
  } = t, n = ht(t, [
    "locale"
  ]);
  return wo(Wr({ locale: r }, n));
}, Ho = (e = {}) => {
  var t = e, {
    locale: r = $e()
  } = t, n = ht(t, [
    "locale"
  ]);
  return To(Wr({ locale: r }, n));
}, Po = (e = {}) => {
  var t = e, {
    locale: r = $e()
  } = t, n = ht(t, [
    "locale"
  ]);
  return So(Wr({ locale: r }, n));
}, Io = tr(
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  (e, t = $e()) => new eo(e, t, ut().formats, {
    ignoreTag: ut().ignoreTag
  })
), Bo = (e, t = {}) => {
  var r, n, i, a;
  let o = t;
  typeof e == "object" && (o = e, e = o.id);
  const {
    values: s,
    locale: u = $e(),
    default: l
  } = o;
  if (u == null)
    throw new Error(
      "[svelte-i18n] Cannot format a message without first setting the initial locale."
    );
  let c = di(e, u);
  if (!c)
    c = (a = (i = (n = (r = ut()).handleMissingMessage) == null ? void 0 : n.call(r, { locale: u, id: e, defaultValue: l })) != null ? i : l) != null ? a : e;
  else if (typeof c != "string")
    return console.warn(
      `[svelte-i18n] Message with id "${e}" must be of type "string", found: "${typeof c}". Gettin its value through the "$format" method is deprecated; use the "json" method instead.`
    ), c;
  if (!s)
    return c;
  let d = c;
  try {
    d = Io(c, u).format(s);
  } catch (v) {
    v instanceof Error && console.warn(
      `[svelte-i18n] Message "${e}" has syntax error:`,
      v.message
    );
  }
  return d;
}, Oo = (e, t) => Po(t).format(e), Lo = (e, t) => Ho(t).format(e), No = (e, t) => Ao(t).format(e), Mo = (e, t = $e()) => di(e, t);
ft([ct, Ht], () => Bo);
ft([ct], () => Oo);
ft([ct], () => Lo);
ft([ct], () => No);
ft([ct, Ht], () => Mo);
const Co = "__i18n__", Ro = [
  "label",
  "info",
  "placeholder",
  "description",
  "title",
  "value"
], Do = [
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
function ko(e) {
  return typeof e == "string" && e.includes(Co);
}
class Uo {
  load_component;
  #t = q(wt({}));
  get shared() {
    return f(this.#t);
  }
  set shared(t) {
    E(this.#t, t, !0);
  }
  #r = q(wt({}));
  get props() {
    return f(this.#r);
  }
  set props(t) {
    E(this.#r, t, !0);
  }
  #e = q((t) => t);
  get i18n() {
    return f(this.#e);
  }
  set i18n(t) {
    E(this.#e, t, !0);
  }
  translatable_props = {};
  dispatcher;
  last_update = null;
  shared_props = Do;
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
    for (const n of Ro)
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
      ), ae(() => {
        this.shared.id = t.shared_props.id;
      });
    }), Object.keys(this.translatable_props).length > 0 && ct.subscribe(() => {
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
      const n = t[r], i = ko(n) ? this._translate_and_store(this.shared_props.includes(r) ? "shared" : "props", r, n) : n;
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
Zt(["click"]);
function br(e) {
  let t = ["", "k", "M", "G", "T", "P", "E", "Z"], r = 0;
  for (; e > 1e3 && r < t.length - 1; )
    e /= 1e3, r++;
  let n = t[r];
  return (Number.isInteger(e) ? e : e.toFixed(1)) + n;
}
function Bn(e) {
  return Object.prototype.toString.call(e) === "[object Date]";
}
function Rr(e, t, r, n) {
  if (typeof r == "number" || Bn(r)) {
    const i = n - r, a = (r - t) / (e.dt || 1 / 60), o = e.opts.stiffness * i, s = e.opts.damping * a, u = (o - s) * e.inv_mass, l = (a + u) * e.dt;
    return Math.abs(l) < e.opts.precision && Math.abs(i) < e.opts.precision ? n : (e.settled = !1, Bn(r) ? new Date(r.getTime() + l) : r + l);
  } else {
    if (Array.isArray(r))
      return r.map(
        (i, a) => (
          // @ts-ignore
          Rr(e, t[a], r[a], n[a])
        )
      );
    if (typeof r == "object") {
      const i = {};
      for (const a in r)
        i[a] = Rr(e, t[a], r[a], n[a]);
      return i;
    } else
      throw new Error(`Cannot spring ${typeof r} values`);
  }
}
function On(e, t = {}) {
  const r = At(e), { stiffness: n = 0.15, damping: i = 0.8, precision: a = 0.01 } = t;
  let o, s, u, l = (
    /** @type {T} */
    e
  ), c = (
    /** @type {T | undefined} */
    e
  ), d = 1, v = 0, w = !1;
  function g(H, h = {}) {
    c = H;
    const m = u = {};
    return e == null || h.hard || x.stiffness >= 1 && x.damping >= 1 ? (w = !0, o = Se.now(), l = H, r.set(e = c), Promise.resolve()) : (h.soft && (v = 1 / ((h.soft === !0 ? 0.5 : +h.soft) * 60), d = 0), s || (o = Se.now(), w = !1, s = Na((p) => {
      if (w)
        return w = !1, s = null, !1;
      d = Math.min(d + v, 1);
      const b = Math.min(p - o, 1e3 / 30), y = {
        inv_mass: d,
        opts: x,
        settled: !0,
        dt: b * 60 / 1e3
      }, A = Rr(y, l, e, c);
      return o = p, l = /** @type {T} */
      e, r.set(e = /** @type {T} */
      A), y.settled && (s = null), !y.settled;
    })), new Promise((p) => {
      s.promise.then(() => {
        m === u && p();
      });
    }));
  }
  const x = {
    set: g,
    update: (H, h) => g(H(
      /** @type {T} */
      c,
      /** @type {T} */
      e
    ), h),
    subscribe: r.subscribe,
    stiffness: n,
    damping: i,
    precision: a
  };
  return x;
}
var Fo = /* @__PURE__ */ se('<div><svg viewBox="-1200 -1200 3000 3000" fill="none" xmlns="http://www.w3.org/2000/svg" class="svelte-m6d381"><g><path d="M255.926 0.754768L509.702 139.936V221.027L255.926 81.8465V0.754768Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M509.69 139.936L254.981 279.641V361.255L509.69 221.55V139.936Z" fill="#FF7C00" class="svelte-m6d381"></path><path d="M0.250138 139.937L254.981 279.641V361.255L0.250138 221.55V139.937Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M255.923 0.232622L0.236328 139.936V221.55L255.923 81.8469V0.232622Z" fill="#FF7C00" class="svelte-m6d381"></path></g><g><path d="M255.926 141.5L509.702 280.681V361.773L255.926 222.592V141.5Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M509.69 280.679L254.981 420.384V501.998L509.69 362.293V280.679Z" fill="#FF7C00" class="svelte-m6d381"></path><path d="M0.250138 280.681L254.981 420.386V502L0.250138 362.295V280.681Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M255.923 140.977L0.236328 280.68V362.294L255.923 222.591V140.977Z" fill="#FF7C00" class="svelte-m6d381"></path></g></svg></div>');
function Go(e, t) {
  Jt(t, !0);
  const r = () => nn(u, "$top", i), n = () => nn(l, "$bottom", i), [i, a] = xa();
  var o = this && this.__awaiter || function(p, b, y, A) {
    function P(B) {
      return B instanceof y ? B : new y(function(D) {
        D(B);
      });
    }
    return new (y || (y = Promise))(function(B, D) {
      function L(K) {
        try {
          ne(A.next(K));
        } catch (G) {
          D(G);
        }
      }
      function Z(K) {
        try {
          ne(A.throw(K));
        } catch (G) {
          D(G);
        }
      }
      function ne(K) {
        K.done ? B(K.value) : P(K.value).then(L, Z);
      }
      ne((A = A.apply(p, b || [])).next());
    });
  };
  let s = I(t, "margin", 3, !0);
  const u = On([0, 0]), l = On([0, 0]);
  let c = q(!1);
  function d() {
    return o(this, void 0, void 0, function* () {
      yield Promise.all([u.set([125, 140]), l.set([-125, -140])]), yield Promise.all([u.set([-125, 140]), l.set([125, -140])]), yield Promise.all([u.set([-125, 0]), l.set([125, -0])]), yield Promise.all([u.set([125, 0]), l.set([-125, 0])]);
    });
  }
  function v() {
    return o(this, void 0, void 0, function* () {
      yield d(), f(c) || v();
    });
  }
  function w() {
    return o(this, void 0, void 0, function* () {
      yield Promise.all([u.set([125, 0]), l.set([-125, 0])]), v();
    });
  }
  Te(() => (w(), () => {
    E(c, !0);
  }));
  var g = Fo();
  let x;
  var H = te(g), h = te(H), m = j(h);
  Q(() => {
    x = De(g, 1, "svelte-m6d381", null, x, { margin: s() }), Ae(h, `transform: translate(${r()[0] ?? ""}px, ${r()[1] ?? ""}px);`), Ae(m, `transform: translate(${n()[0] ?? ""}px, ${n()[1] ?? ""}px);`);
  }), C(e, g), Yt(), a();
}
var jo = function(e, t, r, n) {
  function i(a) {
    return a instanceof r ? a : new r(function(o) {
      o(a);
    });
  }
  return new (r || (r = Promise))(function(a, o) {
    function s(c) {
      try {
        l(n.next(c));
      } catch (d) {
        o(d);
      }
    }
    function u(c) {
      try {
        l(n.throw(c));
      } catch (d) {
        o(d);
      }
    }
    function l(c) {
      c.done ? a(c.value) : i(c.value).then(s, u);
    }
    l((n = n.apply(e, t || [])).next());
  });
};
let kt = [], _r = !1;
const Vo = typeof window < "u", _i = Vo ? window.requestAnimationFrame : (e) => {
};
function zo(e) {
  return jo(this, arguments, void 0, function* (t, r = !0) {
    if (!(window.__gradio_mode__ === "website" || window.__gradio_mode__ !== "app" && r !== !0)) {
      if (kt.push(t), !_r) _r = !0;
      else return;
      yield va(), _i(() => {
        let n = [0, 0];
        for (let i = 0; i < kt.length; i++) {
          const o = kt[i].getBoundingClientRect();
          (i === 0 || o.top + window.scrollY <= n[0]) && (n[0] = o.top + window.scrollY, n[1] = i);
        }
        window.scrollTo({ top: n[0] - 20, behavior: "smooth" }), _r = !1, kt = [];
      });
    }
  });
}
var Xo = /* @__PURE__ */ se('<div class="validation-error svelte-124hqw6"> <button class="svelte-124hqw6"><!></button></div>'), qo = /* @__PURE__ */ se('<div class="eta-bar svelte-124hqw6"></div>'), Wo = /* @__PURE__ */ se("<!> ", 1), Zo = /* @__PURE__ */ se("<!> <!> <!> <!>", 1), Yo = /* @__PURE__ */ se('<div class="progress-level svelte-124hqw6"><div class="progress-level-inner svelte-124hqw6"><!></div> <div class="progress-bar-wrap svelte-124hqw6"><div class="progress-bar svelte-124hqw6"></div></div></div>'), Jo = /* @__PURE__ */ se('<p class="loading svelte-124hqw6"> </p> <!>', 1), Qo = /* @__PURE__ */ se("<!> <div><!> <!></div> <!> <!>", 1), Ko = /* @__PURE__ */ se('<div class="clear-status svelte-124hqw6"><!></div> <span class="error svelte-124hqw6"> </span> <!>', 1), $o = /* @__PURE__ */ se("<div> <!> </div>"), el = /* @__PURE__ */ se('<div data-testid="status-tracker"><!> <!></div> <!>', 1);
function tl(e, t) {
  Jt(t, !0);
  let r = I(t, "eta", 3, null), n = I(t, "scroll_to_output", 3, !1), i = I(t, "timer", 3, !0), a = I(t, "show_progress", 3, "full"), o = I(t, "message", 3, null), s = I(t, "progress", 3, null), u = I(t, "variant", 3, "default"), l = I(t, "loading_text", 3, "Loading..."), c = I(t, "absolute", 3, !0), d = I(t, "translucent", 3, !1), v = I(t, "border", 3, !1), w = I(t, "validation_error", 7, null), g = I(t, "show_validation_error", 3, !0), x = I(t, "type", 3, null), H = I(t, "used_cache", 3, null), h = I(t, "cache_duration", 3, null), m = I(t, "avg_time", 3, null), p, b = !1, y = q(0), A = q(null), P = q(null), B = q(!1), D = q(null), L = q(!1), Z = q(!1), ne = q(null), K = q(null), G = q("from cache"), ke = q(!1), He = null, Ue = null;
  const pe = we(() => !(g() && w()) && (x() === "input" || !t.status || t.status === "complete" || a() === "hidden" || t.status == "streaming"));
  let Pe = q(0);
  const dt = we(() => f(P) === null || f(P) <= 0 || !f(Pe) ? 0 : Math.min(f(Pe) / f(P), 1)), J = we(() => f(Pe).toFixed(1));
  let de = we(() => s() == null), ve = we(() => r() !== null && r() !== void 0 ? r() : f(A));
  function me() {
    _i(() => {
      E(Pe, (performance.now() - f(y)) / 1e3), b && me();
    });
  }
  let re = we(() => {
    let V = null;
    s() != null ? V = s().map((_) => {
      if (_.index != null && _.length != null)
        return _.index / _.length;
      if (_.progress != null)
        return _.progress;
    }) : V = null;
    let $, F = "";
    return V ? ($ = V[V.length - 1], $ === 0 ? F = "0" : F = "150ms") : $ = void 0, {
      progress_level: V,
      last_progress_level: $,
      progress_bar_transition: F
    };
  });
  function ge() {
    b || (E(A, E(D, null), !0), E(y, performance.now(), !0), b = !0, me());
  }
  function Fe() {
    E(A, E(D, null), !0), b && (b = !1);
  }
  Te(() => {
    t.status === "pending" ? ge() : ae(() => {
      Fe();
    });
  }), Te(() => {
    p && n() && (t.status === "pending" || t.status === "complete") && zo(p, t.autoscroll);
  }), Te(() => {
    f(ve) != null && f(A) !== f(ve) && (E(P, (performance.now() - f(y)) / 1e3 + f(ve)), E(D, f(P).toFixed(1), !0), E(A, f(ve), !0));
  });
  function he() {
    E(B, !1);
  }
  Te(() => {
    ae(() => {
      he();
    }), t.status === "error" && o() && E(B, !0);
  }), Te(() => {
    t.status === "complete" && x() === "output" && H() && h() != null && (E(ne, h().toFixed(1), !0), E(G, H() === "full" ? "from cache" : "used cache", !0), E(ke, m() != null && m() > h() && m() > 0, !0), E(K, f(ke) ? m().toFixed(1) : null, !0), E(L, !0), E(Z, !1), He && clearTimeout(He), Ue && clearTimeout(Ue), He = setTimeout(
      () => {
        E(Z, !0), Ue = setTimeout(
          () => {
            E(L, !1), E(Z, !1);
          },
          500
        );
      },
      1750
    ));
  });
  var Ge = el(), Ee = fe(Ge);
  let qe, be;
  var Le = te(Ee);
  {
    var Pt = (V) => {
      var $ = Xo(), F = te($), _ = j(F), T = te(_);
      {
        let O = we(() => t.i18n ? t.i18n("common.clear") : "Clear");
        pn(T, {
          get Icon() {
            return vn;
          },
          get label() {
            return f(O);
          },
          disabled: !1,
          size: "x-small",
          background: "var(--background-fill-primary)",
          color: "var(--error-background-text)",
          border: "var(--border-color-primary)",
          onclick: () => w(null)
        });
      }
      Q(() => ue(F, `${w() ?? ""} `)), C(V, $);
    };
    Y(Le, (V) => {
      w() && g() && V(Pt);
    });
  }
  var It = j(Le, 2);
  {
    var Bt = (V) => {
      var $ = Qo(), F = fe($);
      {
        var _ = (N) => {
          var W = qo();
          let _e;
          Q(() => _e = Ae(W, "", _e, {
            transform: `translateX(${(f(dt) || 0) * 100 - 100}%)`
          })), C(N, W);
        };
        Y(F, (N) => {
          u() === "default" && f(de) && a() === "full" && N(_);
        });
      }
      var T = j(F, 2);
      let O;
      var S = te(T);
      {
        var U = (N) => {
          var W = it(), _e = fe(W);
          on(_e, 17, s, an, (Ne, ye) => {
            var Nt = it(), ar = fe(Nt);
            {
              var Mt = (We) => {
                var vt = Wo(), Ct = fe(vt);
                {
                  var sr = (Me) => {
                    var Ze = Be();
                    Q((mt, gt) => ue(Ze, `${mt ?? ""}/${gt ?? ""}`), [
                      () => br(f(ye).index || 0),
                      () => br(f(ye).length)
                    ]), C(Me, Ze);
                  }, et = (Me) => {
                    var Ze = Be();
                    Q((mt) => ue(Ze, mt), [() => br(f(ye).index || 0)]), C(Me, Ze);
                  };
                  Y(Ct, (Me) => {
                    f(ye).length != null ? Me(sr) : Me(et, -1);
                  });
                }
                var tt = j(Ct);
                Q(() => ue(tt, ` ${f(ye).unit ?? ""} |  `)), C(We, vt);
              };
              Y(ar, (We) => {
                f(ye).index != null && We(Mt);
              });
            }
            C(Ne, Nt);
          }), C(N, W);
        }, ee = (N) => {
          var W = Be();
          Q(() => ue(W, `queue: ${t.queue_position + 1}/${t.queue_size ?? ""} |`)), C(N, W);
        }, ie = (N) => {
          var W = Be("processing |");
          C(N, W);
        };
        Y(S, (N) => {
          s() ? N(U) : t.queue_position !== null && t.queue_size !== void 0 && t.queue_position >= 0 ? N(ee, 1) : t.queue_position === 0 && N(ie, 2);
        });
      }
      var z = j(S, 2);
      {
        var oe = (N) => {
          var W = Be();
          Q(() => ue(W, `${f(J) ?? ""}${r() ? `/${f(D)}` : ""}s`)), C(N, W);
        };
        Y(z, (N) => {
          i() && N(oe);
        });
      }
      var je = j(T, 2);
      {
        var Ot = (N) => {
          var W = Yo(), _e = te(W), Ne = te(_e);
          {
            var ye = (We) => {
              var vt = it(), Ct = fe(vt);
              on(Ct, 17, s, an, (sr, et, tt) => {
                var Me = it(), Ze = fe(Me);
                {
                  var mt = (gt) => {
                    var Zr = Zo(), Yr = fe(Zr);
                    {
                      var yi = (le) => {
                        var Ce = Be(" /");
                        C(le, Ce);
                      };
                      Y(Yr, (le) => {
                        tt !== 0 && le(yi);
                      });
                    }
                    var Jr = j(Yr, 2);
                    {
                      var xi = (le) => {
                        var Ce = Be();
                        Q(() => ue(Ce, f(et).desc)), C(le, Ce);
                      };
                      Y(Jr, (le) => {
                        f(et).desc != null && le(xi);
                      });
                    }
                    var Qr = j(Jr, 2);
                    {
                      var Ei = (le) => {
                        var Ce = Be("-");
                        C(le, Ce);
                      };
                      Y(Qr, (le) => {
                        f(et).desc != null && f(re).progress_level && f(re).progress_level[tt] != null && le(Ei);
                      });
                    }
                    var wi = j(Qr, 2);
                    {
                      var Ti = (le) => {
                        var Ce = Be();
                        Q((Si) => ue(Ce, `${Si ?? ""}%`), [
                          () => (100 * (f(re).progress_level[tt] || 0)).toFixed(1)
                        ]), C(le, Ce);
                      };
                      Y(wi, (le) => {
                        f(re).progress_level != null && le(Ti);
                      });
                    }
                    C(gt, Zr);
                  };
                  Y(Ze, (gt) => {
                    (f(et).desc != null || f(re).progress_level && f(re).progress_level[tt] != null) && gt(mt);
                  });
                }
                C(sr, Me);
              }), C(We, vt);
            };
            Y(Ne, (We) => {
              s() != null && We(ye);
            });
          }
          var Nt = j(_e, 2), ar = te(Nt);
          let Mt;
          Q(() => Mt = Ae(ar, "", Mt, {
            width: `${f(re).last_progress_level * 100}%`,
            transition: f(re).progress_bar_transition
          })), C(N, W);
        }, pt = (N) => {
          {
            let W = we(() => u() === "default");
            Go(N, {
              get margin() {
                return f(W);
              }
            });
          }
        };
        Y(je, (N) => {
          f(re).last_progress_level != null ? N(Ot) : a() === "full" && N(pt, 1);
        });
      }
      var ir = j(je, 2);
      {
        var Lt = (N) => {
          var W = Jo(), _e = fe(W), Ne = te(_e), ye = j(_e, 2);
          Sr(ye, t, "additional-loading-text", {}), Q(() => ue(Ne, l())), C(N, W);
        };
        Y(ir, (N) => {
          i() || N(Lt);
        });
      }
      Q(() => O = De(T, 1, "progress-text svelte-124hqw6", null, O, {
        "meta-text-center": u() === "center",
        "meta-text": u() === "default"
      })), C(V, $);
    }, rr = (V) => {
      var $ = Ko(), F = fe($), _ = te(F);
      {
        let U = we(() => t.i18n("common.clear"));
        pn(_, {
          get Icon() {
            return vn;
          },
          get label() {
            return f(U);
          },
          disabled: !1,
          $$events: {
            click: () => {
              t.on_clear_status?.();
            }
          }
        });
      }
      var T = j(F, 2), O = te(T), S = j(T, 2);
      Sr(S, t, "error", {}), Q((U) => ue(O, U), [() => t.i18n("common.error")]), C(V, $);
    };
    Y(It, (V) => {
      t.status === "pending" ? V(Bt) : t.status === "error" && V(rr, 1);
    });
  }
  Xr(Ee, (V) => p = V, () => p);
  var nr = j(Ee, 2);
  {
    var Ie = (V) => {
      var $ = $o();
      let F, _;
      var T = te($), O = j(T);
      {
        var S = (ee) => {
          var ie = Be();
          Q(() => ue(ie, `~${f(K) ?? ""}s
			→ `)), C(ee, ie);
        };
        Y(O, (ee) => {
          f(ke) && ee(S);
        });
      }
      var U = j(O);
      Q(() => {
        F = De($, 1, "cache-indicator svelte-124hqw6", null, F, { "fade-out": f(Z) }), _ = Ae($, "", _, { position: c() ? "absolute" : "static" }), ue(T, `⚡ ${f(G) ?? ""}: `), ue(U, `${f(ne) ?? ""}s`);
      }), C(V, $);
    };
    Y(nr, (V) => {
      f(L) && V(Ie);
    });
  }
  Q(() => {
    qe = De(Ee, 1, `wrap ${u() ?? ""} ${a() ?? ""}`, "svelte-124hqw6", qe, {
      "no-click": w() && g(),
      hide: f(pe),
      translucent: u() === "center" && (t.status === "pending" || t.status === "error") || d() || a() === "minimal" || w(),
      generating: t.status === "generating" && a() === "full",
      border: v()
    }), be = Ae(Ee, "", be, {
      position: c() ? "absolute" : "static",
      padding: c() ? "0" : "var(--size-8) 0"
    });
  }), C(e, Ge), Yt();
}
const rl = (e) => {
  const t = {};
  for (let r = 0, n = e.length; r < n; r++) {
    const i = e[r];
    for (const a in i)
      t[a] ? t[a] = t[a].concat(i[a]) : t[a] = i[a];
  }
  return t;
}, nl = [
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
], il = [
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
], al = [
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
rl([
  Object.fromEntries(nl.map((e) => [e, ["*"]])),
  Object.fromEntries(il.map((e) => [e, ["svg:*"]])),
  Object.fromEntries(al.map((e) => [e, ["math:*"]]))
]);
Zt(["touchstart", "touchmove", "touchend", "click", "keydown"]);
var sl = /* @__PURE__ */ new Set(["$$slots", "$$events", "$$legacy"]), ol = /* @__PURE__ */ se('<!> <div class="region-annotator svelte-r41nsf"><div class="toolbar svelte-r41nsf" role="group" aria-label="Region annotation tool"><button type="button">浏览</button> <button type="button">套索选择</button> <button type="button" class="finish svelte-r41nsf">完成套索</button> <button type="button" class="clear svelte-r41nsf">清除 Draft</button></div> <div class="legend svelte-r41nsf"><span class="svelte-r41nsf"><i class="draft svelte-r41nsf"></i>黄色：Draft</span> <span class="svelte-r41nsf"><i class="saved svelte-r41nsf"></i>绿色：Saved Region</span></div> <div class="canvas-wrap svelte-r41nsf"><canvas class="svelte-r41nsf"></canvas></div> <div class="status svelte-r41nsf"> </div></div>', 1);
function ul(e, t) {
  Jt(t, !0);
  const r = /* @__PURE__ */ Ya(t, sl), n = new Uo(r), i = 2048, a = 4096, o = 3;
  let s, u = null, l = null, c = null, d = null, v = null, w = q(wt({})), g = q("browse"), x = q("请先生成版图 binary mask"), H = q(!1), h = q(!1), m = q(!1), p = q(wt([])), b = null, y = null, A = null, P = !1, B = "", D = 0, L = "", Z = null, ne = !1, K = !1, G = q(!1);
  function ke(_) {
    return JSON.parse(JSON.stringify(_ || {}));
  }
  function He(_) {
    return typeof _ == "number" ? `${_}px` : _ || "520px";
  }
  function Ue(_, T, O) {
    return Math.max(T, Math.min(O, _));
  }
  function pe() {
    return f(w).server_view || {};
  }
  function Pe() {
    return f(w).client_intent || {};
  }
  function dt(_ = Pe()) {
    const T = pe();
    return JSON.stringify([
      _.session_id || "",
      _.layout_id || "",
      _.source_mask_hash || "",
      Number(T.natural_width || 0),
      Number(T.natural_height || 0)
    ]);
  }
  function J() {
    return Math.max(1, Number(pe().natural_width || u?.naturalWidth || 1));
  }
  function de() {
    return Math.max(1, Number(pe().natural_height || u?.naturalHeight || 1));
  }
  function ve() {
    return Math.min(1, i / Math.max(J(), de()));
  }
  function me(_, T, O) {
    const S = (ee) => {
      T === D && O(ee);
    };
    if (!_) {
      S(null);
      return;
    }
    const U = new Image();
    U.onload = () => S(U), U.onerror = () => S(null), U.src = _;
  }
  function re(_, T) {
    _ === D && (E(G, ne && K && u !== null && l !== null, !0), ne && K && E(x, f(G) ? T : "当前 Layout 图像加载失败，套索已禁用", !0), F());
  }
  function ge() {
    if (!l) {
      c = null;
      return;
    }
    const _ = J(), T = de(), O = document.createElement("canvas");
    O.width = _, O.height = T;
    const S = O.getContext("2d", { willReadFrequently: !0 });
    if (!S) return;
    S.imageSmoothingEnabled = !1, S.drawImage(l, 0, 0, _, T);
    const U = S.getImageData(0, 0, _, T), ee = document.createElement("canvas");
    ee.width = _, ee.height = T;
    const ie = ee.getContext("2d");
    if (!ie) return;
    const z = ie.createImageData(_, T);
    for (let oe = 0; oe < U.data.length; oe += 4) {
      const je = Math.max(U.data[oe], U.data[oe + 1], U.data[oe + 2]);
      U.data[oe + 3] > 0 && je >= 128 && (z.data[oe] = 45, z.data[oe + 1] = 160, z.data[oe + 2] = 255, z.data[oe + 3] = 80);
    }
    ie.putImageData(z, 0, 0), c = ee;
  }
  function Fe(_) {
    E(w, ke(_), !0);
    const T = Pe(), O = pe(), S = ++D, U = dt(T), ee = U !== L;
    L = U;
    const ie = O.status || "Region 标注器已加载";
    if (E(g, T.tool_mode === "lasso" ? "lasso" : "browse", !0), E(
      p,
      Array.isArray(T.lasso_polygon) ? T.lasso_polygon.filter((z) => Array.isArray(z) && z.length === 2).map((z) => ({ x: Number(z[0]), y: Number(z[1]) })) : [],
      !0
    ), b !== null && s?.hasPointerCapture(b) && s.releasePointerCapture(b), E(H, !1), E(h, !1), E(m, !1), b = null, Z = null, y = null, A = null, P = !1, d = null, v = null, ee || O.enabled !== !0 ? (u = null, l = null, c = null, ne = !1, K = !1, E(G, !1)) : (ne = u !== null, K = l !== null, E(G, ne && K, !0)), O.enabled !== !0) {
      E(x, ie, !0), F();
      return;
    }
    E(x, f(G) ? ie : "正在加载当前 Layout 图像…", !0), F(), me(O.source_image, S, (z) => {
      u = z, ne = !0, re(S, ie);
    }), me(O.source_mask_image, S, (z) => {
      l = z, K = !0, ge(), re(S, ie);
    }), me(O.saved_region_overlay_image, S, (z) => {
      d = z, F();
    }), me(O.draft_region_overlay_image, S, (z) => {
      v = z, F();
    });
  }
  Te(() => {
    const _ = JSON.stringify(n.props.value || null);
    _ !== B && (B = _, Fe(n.props.value));
  });
  function he(_ = !1) {
    E(
      w,
      Object.assign(Object.assign({}, f(w)), {
        client_intent: Object.assign(Object.assign({}, Pe()), {
          tool_mode: f(g),
          lasso_polygon: f(p).map((T) => [T.x, T.y])
        })
      }),
      !0
    ), n.props.value = f(w), B = JSON.stringify(f(w)), _ && n.dispatch("input");
  }
  function Ge(_) {
    (f(H) || f(h)) && Ie("绘制已取消"), E(g, _, !0), E(
      x,
      _ === "lasso" ? "按住拖动绘制自由线，松开后可继续点击添加直线段；点击完成后统一闭合" : "浏览模式：Canvas 只读",
      !0
    ), he(!1), F();
  }
  function Ee(_) {
    const T = s.getBoundingClientRect();
    return {
      x: Ue((_.clientX - T.left) / Math.max(1, T.width) * J(), 0, J() - 1),
      y: Ue((_.clientY - T.top) / Math.max(1, T.height) * de(), 0, de() - 1)
    };
  }
  function qe() {
    E(
      w,
      Object.assign(Object.assign({}, f(w)), {
        server_view: Object.assign(Object.assign({}, pe()), { draft_region_overlay_image: "" })
      }),
      !0
    ), v = null;
  }
  function be(_) {
    if (_.button !== 0 || f(g) !== "lasso" || pe().enabled !== !0) return;
    if (!f(G)) {
      E(x, "当前 Layout 图像尚未加载完成，不能开始套索"), F();
      return;
    }
    if (_.preventDefault(), f(h) && Z !== L) {
      Ie("Layout 已切换，Draft 已清除");
      return;
    }
    const T = Ee(_);
    f(h) || (qe(), E(p, [T], !0), Z = L, E(m, !1)), E(H, !0), P = !1, b = _.pointerId, A = T, y = { x: _.clientX, y: _.clientY }, s.setPointerCapture(_.pointerId), E(x, f(h) ? "松开添加直线顶点，或继续拖动绘制自由线条" : "正在绘制 Draft", !0), he(!1), F();
  }
  function Le(_) {
    if (!f(H) || _.pointerId !== b || !y) return;
    if (Z !== L) {
      Ie("Layout 已切换，Draft 已清除");
      return;
    }
    if (_.preventDefault(), Math.hypot(_.clientX - y.x, _.clientY - y.y) < o) return;
    const O = Ee(_);
    if (!P) {
      if (P = !0, f(h) && A) {
        const S = f(p)[f(p).length - 1];
        (!S || S.x !== A.x || S.y !== A.y) && f(p).length < a && E(p, [...f(p), A], !0);
      }
      E(h, !1);
    }
    f(p).length < a ? E(p, [...f(p), O], !0) : (E(p, [...f(p).slice(0, -1), O], !0), E(m, !0)), y = { x: _.clientX, y: _.clientY }, E(
      x,
      f(m) ? `已达到 ${a} 点上限` : `Draft 点数：${f(p).length}`,
      !0
    ), F();
  }
  function Pt(_) {
    const T = Ee(_), O = f(p)[f(p).length - 1];
    O && O.x === T.x && O.y === T.y || (f(p).length < a ? E(p, [...f(p), T], !0) : (E(p, [...f(p).slice(0, -1), T], !0), E(m, !0)));
  }
  function It(_, T = !1) {
    b !== null && s.hasPointerCapture(b) && s.releasePointerCapture(b), b = null, T || (Z = null), y = null, A = null, E(H, !1), P = !1, _.preventDefault();
  }
  function Bt() {
    return new Set(f(p).map((_) => `${_.x.toFixed(4)},${_.y.toFixed(4)}`)).size;
  }
  function rr() {
    if (f(h)) {
      if (Z !== L) {
        Ie("Layout 已切换，Draft 已清除");
        return;
      }
      if (f(p).length < 3 || Bt() < 3) {
        E(x, "套索至少需要 3 个不同点"), F();
        return;
      }
      E(h, !1), Z = null, E(x, "正在生成权威 Draft 交集预览…"), he(!0), F();
    }
  }
  function nr(_) {
    if (!f(H) || _.pointerId !== b) return;
    if (Z !== L) {
      Ie("Layout 已切换，Draft 已清除");
      return;
    }
    if (P) {
      Pt(_), It(_, !0), E(h, !0), E(
        x,
        f(m) ? `已达到 ${a} 点上限，请点击“完成套索”` : "自由线段已保留；可继续拖动或点击添加直线段，最后点击“完成套索”",
        !0
      ), he(!1), F();
      return;
    }
    f(h) && Pt(_), It(_, !0), E(h, !0), E(
      x,
      f(m) ? `已达到 ${a} 点上限，请完成套索` : `Draft 点数：${f(p).length}；可继续拖动或点击添加直线段，最后点击“完成套索”`,
      !0
    ), he(!1), F();
  }
  function Ie(_ = "Draft 已清除") {
    b !== null && s?.hasPointerCapture(b) && s.releasePointerCapture(b), b = null, Z = null, y = null, A = null, E(H, !1), E(h, !1), P = !1, E(m, !1), E(p, [], !0), qe(), E(x, _, !0), he(!1), F();
  }
  function V(_) {
    _.pointerId === b && Ie("指针操作已取消，Draft 已清除");
  }
  function $() {
    (f(H) || f(h)) && Ie("窗口失焦，Draft 已清除");
  }
  function F() {
    if (!s) return;
    const _ = J(), T = de(), O = ve();
    s.width = Math.max(1, Math.round(_ * O)), s.height = Math.max(1, Math.round(T * O));
    const S = s.getContext("2d");
    if (S && (S.setTransform(O, 0, 0, O, 0, 0), S.clearRect(0, 0, _, T), u ? (S.imageSmoothingEnabled = !0, S.drawImage(u, 0, 0, _, T)) : (S.fillStyle = "#f8fafc", S.fillRect(0, 0, _, T), S.fillStyle = "#64748b", S.font = "18px sans-serif", S.fillText(pe().enabled === !0 ? "正在加载当前 Layout 图像…" : "请先生成版图 binary mask", 24, 42)), S.imageSmoothingEnabled = !1, c && S.drawImage(c, 0, 0, _, T), d && S.drawImage(d, 0, 0, _, T), v && S.drawImage(v, 0, 0, _, T), f(p).length > 0)) {
      S.save(), S.beginPath(), S.moveTo(f(p)[0].x, f(p)[0].y);
      for (let U = 1; U < f(p).length; U += 1) S.lineTo(f(p)[U].x, f(p)[U].y);
      if (!f(H) && !f(h) && f(p).length >= 3 && S.closePath(), S.setLineDash([8, 5]), S.lineWidth = Math.max(2, _ / 700), S.strokeStyle = "rgba(255, 220, 0, 0.98)", S.stroke(), f(h)) {
        S.setLineDash([]), S.fillStyle = "rgba(255, 220, 0, 0.98)";
        const U = Math.max(2.5, _ / 500);
        for (const ee of f(p))
          S.beginPath(), S.arc(ee.x, ee.y, U, 0, Math.PI * 2), S.fill();
      }
      S.restore();
    }
  }
  Oe("blur", ma, $);
  {
    let _ = we(() => f(H) ? "focus" : "base");
    es(e, {
      get visible() {
        return n.shared.visible;
      },
      variant: "solid",
      get border_mode() {
        return f(_);
      },
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
      children: (T, O) => {
        var S = ol(), U = fe(S);
        tl(U, Qa(
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
        var ee = j(U, 2), ie = te(ee), z = te(ie);
        let oe;
        var je = j(z, 2);
        let Ot;
        var pt = j(je, 2), ir = j(pt, 2), Lt = j(ie, 4), N = te(Lt);
        Xr(N, (Ne) => s = Ne, () => s);
        var W = j(Lt, 2), _e = te(W);
        Q(
          (Ne, ye) => {
            Ae(ee, Ne), oe = De(z, 1, "svelte-r41nsf", null, oe, { active: f(g) === "browse" }), Ot = De(je, 1, "svelte-r41nsf", null, Ot, { active: f(g) === "lasso" }), pt.disabled = ye, Ae(N, `cursor:${f(g) === "lasso" ? "crosshair" : "default"}`), ue(_e, `${f(x) ?? ""} · 点数 ${f(p).length ?? ""}/4096`);
          },
          [
            () => `min-height:${He(n.props.height)}`,
            () => !f(h) || Bt() < 3
          ]
        ), Oe("click", z, () => Ge("browse")), Oe("click", je, () => Ge("lasso")), Oe("click", pt, rr), Oe("click", ir, () => Ie()), Oe("pointerdown", N, be), Oe("pointermove", N, Le), Oe("pointerup", N, nr), Oe("pointercancel", N, V), C(T, S);
      },
      $$slots: { default: !0 }
    });
  }
  Yt();
}
export {
  ul as default
};

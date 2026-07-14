import { i as Ir, g as An, o as Ai, n as je, u as te, s as Hi, r as dr, m as Ze, a as w, b as h, t as Or, d as Pi, q as Bi, c as Hn, e as Je, f as jt, h as kt, j as Ii, T as Oi, k as Mi, l as Ut, p as Ye, v as Mr, w as Qe, x as Pn, y as Bn, z as In, A as St, E as Vt, B as qr, C as Ni, D as On, F as Nr, G as Li, H as Wr, I as Ci, J as Ri, K as Ce, L as Mn, M as er, N as Di, O as ki, P as Ui, Q as Fi, R as Nn, S as Lr, U as Zr, V as Yr, W as Gi, X as ji, Y as Vi, Z as zi, _ as Xi, $ as qi, a0 as Wi, a1 as Zi, a2 as Cr, a3 as Yi, a4 as Ln, a5 as zt, a6 as Ji, a7 as Qi, a8 as Ki, a9 as $i, aa as Cn, ab as ea, ac as ta, ad as Rr, ae as ra, af as be, ag as na, ah as pe, ai as pr, aj as vr, ak as ia, al as aa, am as wt, an as sa, ao as oa, ap as la, aq as ua, ar as fa, as as ha, at as Rn, au as bt, av as ca, aw as Jr, ax as da, ay as fe, az as Xt, aA as qt, aB as F, aC as Le, aD as W, aE as pa, aF as Y, aG as ue, aH as ge, aI as X, aJ as va, aK as ma } from "./render-DNiZdw6o.js";
const ga = [];
function ba(e, t = !1, r = !1) {
  return Ct(e, /* @__PURE__ */ new Map(), "", ga, null, r);
}
function Ct(e, t, r, n, i = null, a = !1) {
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
    if (Ir(e)) {
      var s = (
        /** @type {Snapshot<any>} */
        Array(e.length)
      );
      t.set(e, s), i !== null && t.set(i, s);
      for (var u = 0; u < e.length; u += 1) {
        var l = e[u];
        u in e && (s[u] = Ct(l, t, r, n, null, a));
      }
      return s;
    }
    if (An(e) === Ai) {
      s = {}, t.set(e, s), i !== null && t.set(i, s);
      for (var f of Object.keys(e))
        s[f] = Ct(
          // @ts-expect-error
          e[f],
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
      return Ct(
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
function Dr(e, t, r) {
  if (e == null)
    return t(void 0), r && r(void 0), je;
  const n = te(
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
function At(e, t = je) {
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
  function o(s, u = je) {
    const l = [s, u];
    return n.add(l), n.size === 1 && (r = t(i, a) || je), s(
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
    let f = 0, p = je;
    const v = () => {
      if (f)
        return;
      p();
      const m = t(n ? l[0] : l, o, s);
      a ? o(m) : p = typeof m == "function" ? m : je;
    }, E = i.map(
      (m, y) => Dr(
        m,
        (T) => {
          l[y] = T, f &= ~(1 << y), u && v();
        },
        () => {
          f |= 1 << y;
        }
      )
    );
    return u = !0, v(), function() {
      dr(E), p(), u = !1;
    };
  });
}
function ya(e) {
  let t;
  return Dr(e, (r) => t = r)(), t;
}
let Mt = !1, mr = /* @__PURE__ */ Symbol("unmounted");
function Qr(e, t, r) {
  const n = r[t] ??= {
    store: null,
    source: Ze(void 0),
    unsubscribe: je
  };
  if (n.store !== e && !(mr in r))
    if (n.unsubscribe(), n.store = e ?? null, e == null)
      n.source.v = void 0, n.unsubscribe = je;
    else {
      var i = !0;
      n.unsubscribe = Dr(e, (a) => {
        i ? n.source.v = a : w(n.source, a);
      }), i = !1;
    }
  return e && mr in r ? ya(e) : h(n.source);
}
function xa() {
  const e = {};
  function t() {
    Or(() => {
      for (var r in e)
        e[r].unsubscribe();
      Pi(e, mr, {
        enumerable: !1,
        value: !0
      });
    });
  }
  return [e, t];
}
function Ea(e) {
  var t = Mt;
  try {
    return Mt = !1, [e(), Mt];
  } finally {
    Mt = t;
  }
}
function wa(e, t) {
  if (t) {
    const r = document.body;
    e.autofocus = !0, Bi(() => {
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
function Dn(e) {
  var t = Hn("template");
  return t.innerHTML = Sa(e.replaceAll("<!>", "<!---->")), t.content;
}
function st(e, t) {
  var r = (
    /** @type {Effect} */
    jt
  );
  r.nodes === null && (r.nodes = { start: e, end: t, a: null, t: null });
}
// @__NO_SIDE_EFFECTS__
function re(e, t) {
  var r = (t & Oi) !== 0, n = (t & Mi) !== 0, i, a = !e.startsWith("<!>");
  return () => {
    i === void 0 && (i = Dn(a ? e : "<!>" + e), r || (i = /** @type {TemplateNode} */
    kt(i)));
    var o = (
      /** @type {TemplateNode} */
      n || Ii ? document.importNode(i, !0) : i.cloneNode(!0)
    );
    if (r) {
      var s = (
        /** @type {TemplateNode} */
        kt(o)
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
        Dn(i)
      ), s = (
        /** @type {Element} */
        kt(o)
      );
      a = /** @type {Element} */
      kt(s);
    }
    var u = (
      /** @type {TemplateNode} */
      a.cloneNode(!0)
    );
    return st(u, u), u;
  };
}
// @__NO_SIDE_EFFECTS__
function kn(e, t) {
  return /* @__PURE__ */ Aa(e, t, "svg");
}
function He(e = "") {
  {
    var t = Je(e + "");
    return st(t, t), t;
  }
}
function it() {
  var e = document.createDocumentFragment(), t = document.createComment(""), r = Je();
  return e.append(t, r), st(t, r), e;
}
function L(e, t) {
  e !== null && e.before(
    /** @type {Node} */
    t
  );
}
class Wt {
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
        Ut(n), this.#n.delete(r);
      else {
        var i = this.#e.get(r);
        i && (Ut(i.effect), this.#r.set(r, i.effect), this.#e.delete(r), i.fragment.lastChild.remove(), this.anchor.before(i.fragment), n = i.effect);
      }
      for (const [a, o] of this.#t) {
        if (this.#t.delete(a), a === t)
          break;
        const s = this.#e.get(o);
        s && (Ye(s.effect), this.#e.delete(o));
      }
      for (const [a, o] of this.#r) {
        if (a === r || this.#n.has(a)) continue;
        const s = () => {
          if (Array.from(this.#t.values()).includes(a)) {
            var l = document.createDocumentFragment();
            Bn(o, l), l.append(Je()), this.#e.set(a, { effect: o, fragment: l });
          } else
            Ye(o);
          this.#n.delete(a), this.#r.delete(a);
        };
        this.#i || !n ? (this.#n.add(a), Mr(o, s, !1)) : s();
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
      r.includes(n) || (Ye(i.effect), this.#e.delete(n));
  };
  /**
   *
   * @param {any} key
   * @param {null | ((target: TemplateNode) => void)} fn
   */
  ensure(t, r) {
    var n = (
      /** @type {Batch} */
      Pn
    ), i = In();
    if (r && !this.#r.has(t) && !this.#e.has(t))
      if (i) {
        var a = document.createDocumentFragment(), o = Je();
        a.append(o), this.#e.set(t, {
          effect: Qe(() => r(o)),
          fragment: a
        });
      } else
        this.#r.set(
          t,
          Qe(() => r(this.anchor))
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
  var n = new Wt(e);
  St(() => {
    const i = t() ?? null;
    n.ensure(i, i && ((a) => i(a, ...r)));
  }, Vt);
}
function z(e, t, r = !1) {
  var n = new Wt(e), i = r ? Vt : 0;
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
function Kr(e, t) {
  return t;
}
function Pa(e, t, r) {
  for (var n = [], i = t.length, a, o = t.length, s = 0; s < i; s++) {
    let p = t[s];
    Mr(
      p,
      () => {
        if (a) {
          if (a.pending.delete(p), a.done.add(p), a.pending.size === 0) {
            var v = (
              /** @type {Set<EachOutroGroup>} */
              e.outrogroups
            );
            gr(e, Nr(a.done)), v.delete(a), v.size === 0 && (e.outrogroups = null);
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
      ), f = (
        /** @type {Element} */
        l.parentNode
      );
      ki(f), f.append(l), e.items.clear();
    }
    gr(e, t, !u);
  } else
    a = {
      pending: new Set(t),
      done: /* @__PURE__ */ new Set()
    }, (e.outrogroups ??= /* @__PURE__ */ new Set()).add(a);
}
function gr(e, t, r = !0) {
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
      a.f |= Ce;
      const o = document.createDocumentFragment();
      Bn(a, o);
    } else
      Ye(t[i], r);
  }
}
var $r;
function en(e, t, r, n, i, a = null) {
  var o = e, s = /* @__PURE__ */ new Map(), u = null, l = On(() => {
    var d = r();
    return (
      /** @type {V[]} */
      Ir(d) ? d : d == null ? [] : Nr(d)
    );
  }), f, p = /* @__PURE__ */ new Map(), v = !0;
  function E(d) {
    (T.effect.f & Mn) === 0 && (T.pending.delete(d), T.fallback = u, Ba(T, f, o, t, n), u !== null && (f.length === 0 ? (u.f & Ce) === 0 ? Ut(u) : (u.f ^= Ce, xt(u, null, o)) : Mr(u, () => {
      u = null;
    })));
  }
  function m(d) {
    T.pending.delete(d);
  }
  var y = St(() => {
    f = /** @type {V[]} */
    h(l);
    for (var d = f.length, c = /* @__PURE__ */ new Set(), _ = (
      /** @type {Batch} */
      Pn
    ), g = In(), b = 0; b < d; b += 1) {
      var H = f[b], P = n(H, b), B = v ? null : s.get(P);
      B ? (B.v && qr(B.v, H), B.i && qr(B.i, b), g && _.unskip_effect(B.e)) : (B = Ia(
        s,
        v ? o : $r ??= Je(),
        H,
        P,
        b,
        i,
        t,
        r
      ), v || (B.e.f |= Ce), s.set(P, B)), c.add(P);
    }
    if (d === 0 && a && !u && (v ? u = Qe(() => a(o)) : (u = Qe(() => a($r ??= Je())), u.f |= Ce)), d > c.size && Ni(), !v)
      if (p.set(_, c), g) {
        for (const [M, D] of s)
          c.has(M) || _.skip_effect(D.e);
        _.oncommit(E), _.ondiscard(m);
      } else
        E(_);
    h(l);
  }), T = { effect: y, items: s, pending: p, outrogroups: null, fallback: u };
  v = !1;
}
function _t(e) {
  for (; e !== null && (e.f & Di) === 0; )
    e = e.next;
  return e;
}
function Ba(e, t, r, n, i) {
  var a = t.length, o = e.items, s = _t(e.effect.first), u, l = null, f = [], p = [], v, E, m, y;
  for (y = 0; y < a; y += 1) {
    if (v = t[y], E = i(v, y), m = /** @type {EachItem} */
    o.get(E).e, e.outrogroups !== null)
      for (const B of e.outrogroups)
        B.pending.delete(m), B.done.delete(m);
    if ((m.f & er) !== 0 && Ut(m), (m.f & Ce) !== 0)
      if (m.f ^= Ce, m === s)
        xt(m, null, r);
      else {
        var T = l ? l.next : s;
        m === e.effect.last && (e.effect.last = m.prev), m.prev && (m.prev.next = m.next), m.next && (m.next.prev = m.prev), Fe(e, l, m), Fe(e, m, T), xt(m, T, r), l = m, f = [], p = [], s = _t(l.next);
        continue;
      }
    if (m !== s) {
      if (u !== void 0 && u.has(m)) {
        if (f.length < p.length) {
          var d = p[0], c;
          l = d.prev;
          var _ = f[0], g = f[f.length - 1];
          for (c = 0; c < f.length; c += 1)
            xt(f[c], d, r);
          for (c = 0; c < p.length; c += 1)
            u.delete(p[c]);
          Fe(e, _.prev, g.next), Fe(e, l, _), Fe(e, g, d), s = d, l = g, y -= 1, f = [], p = [];
        } else
          u.delete(m), xt(m, s, r), Fe(e, m.prev, m.next), Fe(e, m, l === null ? e.effect.first : l.next), Fe(e, l, m), l = m;
        continue;
      }
      for (f = [], p = []; s !== null && s !== m; )
        (u ??= /* @__PURE__ */ new Set()).add(s), p.push(s), s = _t(s.next);
      if (s === null)
        continue;
    }
    (m.f & Ce) === 0 && f.push(m), l = m, s = _t(m.next);
  }
  if (e.outrogroups !== null) {
    for (const B of e.outrogroups)
      B.pending.size === 0 && (gr(e, Nr(B.done)), e.outrogroups?.delete(B));
    e.outrogroups.size === 0 && (e.outrogroups = null);
  }
  if (s !== null || u !== void 0) {
    var b = [];
    if (u !== void 0)
      for (m of u)
        (m.f & er) === 0 && b.push(m);
    for (; s !== null; )
      (s.f & er) === 0 && s !== e.fallback && b.push(s), s = _t(s.next);
    var H = b.length;
    if (H > 0) {
      var P = null;
      Pa(e, b, P);
    }
  }
}
function Ia(e, t, r, n, i, a, o, s) {
  var u = (o & Ci) !== 0 ? (o & Ri) === 0 ? Ze(r, !1, !1) : Wr(r) : null, l = (o & Li) !== 0 ? Wr(i) : null;
  return {
    v: u,
    i: l,
    e: Qe(() => (a(t, u ?? r, l ?? i, s), () => {
      e.delete(n);
    }))
  };
}
function xt(e, t, r) {
  if (e.nodes)
    for (var n = e.nodes.start, i = e.nodes.end, a = t && (t.f & Ce) === 0 ? (
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
function Fe(e, t, r) {
  t === null ? e.effect.first = r : t.next = r, r === null ? e.effect.last = t : r.prev = t;
}
function br(e, t, r, n, i) {
  var a = t.$$slots?.[r], o = !1;
  a === !0 && (a = t[r === "default" ? "children" : r], o = !0), a === void 0 || a(e, o ? () => n : n);
}
function Oa(e, t, r) {
  var n = new Wt(e);
  St(() => {
    var i = t() ?? null;
    n.ensure(i, i && ((a) => r(a, i)));
  }, Vt);
}
const Ma = () => performance.now(), _e = {
  // don't access requestAnimationFrame eagerly outside method
  // this allows basic testing of user code without JSDOM
  // bunder will eval and remove ternary when the user's app is built
  tick: (
    /** @param {any} _ */
    (e) => requestAnimationFrame(e)
  ),
  now: () => Ma(),
  tasks: /* @__PURE__ */ new Set()
};
function Un() {
  const e = _e.now();
  _e.tasks.forEach((t) => {
    t.c(e) || (_e.tasks.delete(t), t.f());
  }), _e.tasks.size !== 0 && _e.tick(Un);
}
function Na(e) {
  let t;
  return _e.tasks.size === 0 && _e.tick(Un), {
    promise: new Promise((r) => {
      _e.tasks.add(t = { c: e, f: r });
    }),
    abort() {
      _e.tasks.delete(t);
    }
  };
}
function La(e, t, r, n, i, a) {
  var o = null, s = (
    /** @type {TemplateNode} */
    e
  ), u = new Wt(s, !1);
  St(() => {
    const l = t() || null;
    var f = l === "svg" ? Fi : void 0;
    if (l === null) {
      u.ensure(null, null);
      return;
    }
    return u.ensure(l, (p) => {
      if (l) {
        if (o = Hn(l, f), st(o, o), n) {
          var v = null, E = o.appendChild(Je());
          n(o, E), v?.remove();
        }
        jt.nodes.end = o, p.before(o);
      }
    }), () => {
    };
  }, Vt), Or(() => {
  });
}
function Ca(e, t) {
  var r = void 0, n;
  Nn(() => {
    r !== (r = t()) && (n && (Ye(n), n = null), r && (n = Qe(() => {
      Lr(() => (
        /** @type {(node: Element) => void} */
        r(e)
      ));
    })));
  });
}
function Fn(e) {
  var t, r, n = "";
  if (typeof e == "string" || typeof e == "number") n += e;
  else if (typeof e == "object") if (Array.isArray(e)) {
    var i = e.length;
    for (t = 0; t < i; t++) e[t] && (r = Fn(e[t])) && (n && (n += " "), n += r);
  } else for (r in e) e[r] && (n && (n += " "), n += r);
  return n;
}
function Ra() {
  for (var e, t, r = 0, n = "", i = arguments.length; r < i; r++) (e = arguments[r]) && (t = Fn(e)) && (n && (n += " "), n += t);
  return n;
}
function Da(e) {
  return typeof e == "object" ? Ra(e) : e ?? "";
}
const tn = Array.from(" \t\n\r\f \v\uFEFF");
function ka(e, t, r) {
  var n = e == null ? "" : "" + e;
  if (t && (n = n ? n + " " + t : t), r) {
    for (var i of Object.keys(r))
      if (r[i])
        n = n ? n + " " + i : i;
      else if (n.length)
        for (var a = i.length, o = 0; (o = n.indexOf(i, o)) >= 0; ) {
          var s = o + a;
          (o === 0 || tn.includes(n[o - 1])) && (s === n.length || tn.includes(n[s])) ? n = (o === 0 ? "" : n.substring(0, o)) + n.substring(s + 1) : o = s;
        }
  }
  return n === "" ? null : n;
}
function rn(e, t = !1) {
  var r = t ? " !important;" : ";", n = "";
  for (var i of Object.keys(e)) {
    var a = e[i];
    a != null && a !== "" && (n += " " + i + ": " + a + r);
  }
  return n;
}
function tr(e) {
  return e[0] !== "-" || e[1] !== "-" ? e.toLowerCase() : e;
}
function Ua(e, t) {
  if (t) {
    var r = "", n, i;
    if (Array.isArray(t) ? (n = t[0], i = t[1]) : n = t, e) {
      e = String(e).replaceAll(/\s*\/\*.*?\*\/\s*/g, "").trim();
      var a = !1, o = 0, s = !1, u = [];
      n && u.push(...Object.keys(n).map(tr)), i && u.push(...Object.keys(i).map(tr));
      var l = 0, f = -1;
      const y = e.length;
      for (var p = 0; p < y; p++) {
        var v = e[p];
        if (s ? v === "/" && e[p - 1] === "*" && (s = !1) : a ? a === v && (a = !1) : v === "/" && e[p + 1] === "*" ? s = !0 : v === '"' || v === "'" ? a = v : v === "(" ? o++ : v === ")" && o--, !s && a === !1 && o === 0) {
          if (v === ":" && f === -1)
            f = p;
          else if (v === ";" || p === y - 1) {
            if (f !== -1) {
              var E = tr(e.substring(l, f).trim());
              if (!u.includes(E)) {
                v !== ";" && p++;
                var m = e.substring(l, p).trim();
                r += " " + m + ";";
              }
            }
            l = p + 1, f = -1;
          }
        }
      }
    }
    return n && (r += rn(n)), i && (r += rn(i, !0)), r = r.trim(), r === "" ? null : r;
  }
  return e == null ? null : String(e);
}
function Re(e, t, r, n, i, a) {
  var o = (
    /** @type {any} */
    e[Zr]
  );
  if (o !== r || o === void 0) {
    var s = ka(r, n, a);
    s == null ? e.removeAttribute("class") : t ? e.className = s : e.setAttribute("class", s), e[Zr] = r;
  } else if (a && i !== a)
    for (var u in a) {
      var l = !!a[u];
      (i == null || l !== !!i[u]) && e.classList.toggle(u, l);
    }
  return a;
}
function rr(e, t = {}, r, n) {
  for (var i in r) {
    var a = r[i];
    t[i] !== a && (r[i] == null ? e.style.removeProperty(i) : e.style.setProperty(i, a, n));
  }
}
function ye(e, t, r, n) {
  var i = (
    /** @type {any} */
    e[Yr]
  );
  if (i !== t) {
    var a = Ua(t, n);
    a == null ? e.removeAttribute("style") : e.style.cssText = a, e[Yr] = t;
  } else n && (Array.isArray(n) ? (rr(e, r?.[0], n[0]), rr(e, r?.[1], n[1], "important")) : rr(e, r, n));
  return n;
}
function _r(e, t, r = !1) {
  if (e.multiple) {
    if (t == null)
      return;
    if (!Ir(t))
      return Gi();
    for (var n of e.options)
      n.selected = t.includes(nn(n));
    return;
  }
  for (n of e.options) {
    var i = nn(n);
    if (ji(i, t)) {
      n.selected = !0;
      return;
    }
  }
  (!r || t !== void 0) && (e.selectedIndex = -1);
}
function Fa(e) {
  var t = new MutationObserver(() => {
    _r(e, e.__value);
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
  }), Or(() => {
    t.disconnect();
  });
}
function nn(e) {
  return "__value" in e ? e.__value : e.value;
}
const Et = /* @__PURE__ */ Symbol("class"), nt = /* @__PURE__ */ Symbol("style"), Gn = /* @__PURE__ */ Symbol("is custom element"), jn = /* @__PURE__ */ Symbol("is html"), Ga = Cr ? "input" : "INPUT", ja = Cr ? "option" : "OPTION", Va = Cr ? "select" : "SELECT";
function za(e, t) {
  t ? e.hasAttribute("selected") || e.setAttribute("selected", "") : e.removeAttribute("selected");
}
function at(e, t, r, n) {
  var i = Vn(e);
  i[t] !== (i[t] = r) && (t === "loading" && (e[Vi] = r), r == null ? e.removeAttribute(t) : typeof r != "string" && zn(e).includes(t) ? e[t] = r : e.setAttribute(t, r));
}
function Xa(e, t, r, n, i = !1, a = !1) {
  var o = Vn(e), s = o[Gn], u = !o[jn], l = t || {}, f = e.nodeName === ja;
  for (var p in t)
    p in r || (r[p] = null);
  r.class ? r.class = Da(r.class) : r.class = null, r[nt] && (r.style ??= null);
  var v = zn(e);
  if (e.nodeName === Ga && "type" in r && ("value" in r || "__value" in r)) {
    var E = r.type;
    (E !== l.type || E === void 0 && e.hasAttribute("type")) && (l.type = E, at(e, "type", E));
  }
  for (const g in r) {
    let b = r[g];
    if (f && g === "value" && b == null) {
      e.value = e.__value = "", l[g] = b;
      continue;
    }
    if (g === "class") {
      var m = e.namespaceURI === "http://www.w3.org/1999/xhtml";
      Re(e, m, b, n, t?.[Et], r[Et]), l[g] = b, l[Et] = r[Et];
      continue;
    }
    if (g === "style") {
      ye(e, b, t?.[nt], r[nt]), l[g] = b, l[nt] = r[nt];
      continue;
    }
    var y = l[g];
    if (!(b === y && !(b === void 0 && e.hasAttribute(g)))) {
      l[g] = b;
      var T = g[0] + g[1];
      if (T !== "$$")
        if (T === "on") {
          const H = {}, P = "$$" + g;
          let B = g.slice(2);
          var d = $i(B);
          if (Yi(B) && (B = B.slice(0, -7), H.capture = !0), !d && y) {
            if (b != null) continue;
            e.removeEventListener(B, l[P], H), l[P] = null;
          }
          if (d)
            Ln(B, e, b), zt([B]);
          else if (b != null) {
            let M = function(D) {
              l[g].call(this, D);
            };
            l[P] = Ji(B, e, M, H);
          }
        } else if (g === "style")
          at(e, g, b);
        else if (g === "autofocus")
          wa(
            /** @type {HTMLElement} */
            e,
            !!b
          );
        else if (!s && (g === "__value" || g === "value" && b != null))
          e.value = e.__value = b;
        else if (g === "selected" && f)
          za(
            /** @type {HTMLOptionElement} */
            e,
            b
          );
        else {
          var c = g;
          u || (c = Qi(c));
          var _ = c === "defaultValue" || c === "defaultChecked";
          if (b == null && !s && !_)
            if (o[g] = null, c === "value" || c === "checked") {
              let H = (
                /** @type {HTMLInputElement} */
                e
              );
              const P = t === void 0;
              if (c === "value") {
                let B = H.defaultValue;
                H.removeAttribute(c), H.defaultValue = B, H.value = H.__value = P ? B : null;
              } else {
                let B = H.defaultChecked;
                H.removeAttribute(c), H.defaultChecked = B, H.checked = P ? B : !1;
              }
            } else
              e.removeAttribute(g);
          else _ || v.includes(c) && (s || typeof b != "string") ? (e[c] = b, c in o && (o[c] = Ki)) : typeof b != "function" && at(e, c, b);
        }
    }
  }
  return l;
}
function qa(e, t, r = [], n = [], i = [], a, o = !1, s = !1) {
  Wi(i, r, n, (u) => {
    var l = void 0, f = {}, p = e.nodeName === Va, v = !1;
    if (Nn(() => {
      var m = t(...u.map(h)), y = Xa(
        e,
        l,
        m,
        a,
        o,
        s
      );
      v && p && "value" in m && _r(
        /** @type {HTMLSelectElement} */
        e,
        m.value
      );
      for (let d of Object.getOwnPropertySymbols(f))
        m[d] || Ye(f[d]);
      for (let d of Object.getOwnPropertySymbols(m)) {
        var T = m[d];
        d.description === Zi && (!l || T !== l[d]) && (f[d] && Ye(f[d]), f[d] = Qe(() => Ca(e, () => T))), y[d] = T;
      }
      l = y;
    }), p) {
      var E = (
        /** @type {HTMLSelectElement} */
        e
      );
      Lr(() => {
        _r(
          E,
          /** @type {Record<string | symbol, any>} */
          l.value,
          !0
        ), Fa(E);
      });
    }
    v = !0;
  });
}
function Vn(e) {
  return (
    /** @type {Record<string | symbol, unknown>} **/
    /** @type {any} */
    e[zi] ??= {
      [Gn]: e.nodeName.includes("-"),
      [jn]: e.namespaceURI === Xi
    }
  );
}
var an = /* @__PURE__ */ new Map();
function zn(e) {
  var t = e.getAttribute("is") || e.nodeName, r = an.get(t);
  if (r) return r;
  an.set(t, r = []);
  for (var n, i = e, a = Element.prototype; a !== i; ) {
    n = qi(i);
    for (var o in n)
      n[o].set && // better safe than sorry, we don't want spread attributes to mess with HTML content
      o !== "innerHTML" && o !== "textContent" && o !== "innerText" && r.push(o);
    i = An(i);
  }
  return r;
}
function nr(e, t) {
  return e === t || e?.[Rr] === t;
}
function kr(e = {}, t, r, n) {
  var i = (
    /** @type {ComponentContext} */
    Cn.r
  ), a = (
    /** @type {Effect} */
    jt
  );
  return Lr(() => {
    var o, s;
    return ea(() => {
      o = s, s = [], te(() => {
        nr(r(...s), e) || (t(e, ...s), o && nr(r(...o), e) && t(null, ...o));
      });
    }), () => {
      let u = a;
      for (; u !== i && u.parent !== null && u.parent.f & ta; )
        u = u.parent;
      const l = () => {
        s && nr(r(...s), e) && t(null, ...s);
      }, f = u.teardown;
      u.teardown = () => {
        l(), f?.();
      };
    };
  }), e;
}
function Wa(e = !1) {
  const t = (
    /** @type {ComponentContextLegacy} */
    Cn
  ), r = t.l.u;
  if (!r) return;
  let n = () => pe(t.s);
  if (e) {
    let i = 0, a = (
      /** @type {Record<string, any>} */
      {}
    );
    const o = pr(() => {
      let s = !1;
      const u = t.s;
      for (const l in u)
        u[l] !== a[l] && (a[l] = u[l], s = !0);
      return s && i++, i;
    });
    n = () => h(o);
  }
  r.b.length && ra(() => {
    sn(t, n), dr(r.b);
  }), be(() => {
    const i = te(() => r.m.map(na));
    return () => {
      for (const a of i)
        typeof a == "function" && a();
    };
  }), r.a.length && be(() => {
    sn(t, n), dr(r.a);
  });
}
function sn(e, t) {
  if (e.l.s)
    for (const r of e.l.s) h(r);
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
      const a = vr(i, t);
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
        const i = vr(n, t);
        return i && !i.configurable && (i.configurable = !0), i;
      }
    }
  },
  has(e, t) {
    if (t === Rr || t === Rn) return !1;
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
function S(e, t, r, n) {
  var i = !oa || (r & la) !== 0, a = (r & sa) !== 0, o = (r & fa) !== 0, s = (
    /** @type {V} */
    n
  ), u = !0, l = (
    /** @type {Derived<V> | undefined} */
    void 0
  ), f = () => o && i ? (l ??= pr(
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
    var v = Rr in e || Rn in e;
    p = vr(e, t)?.set ?? (v && t in e ? (g) => e[t] = g : void 0);
  }
  var E, m = !1;
  a ? [E, m] = Ea(() => (
    /** @type {V} */
    e[t]
  )) : E = /** @type {V} */
  e[t], E === void 0 && n !== void 0 && (E = f(), p && (i && ia(), p(E)));
  var y;
  if (i ? y = () => {
    var g = (
      /** @type {V} */
      e[t]
    );
    return g === void 0 ? f() : (u = !0, g);
  } : y = () => {
    var g = (
      /** @type {V} */
      e[t]
    );
    return g !== void 0 && (s = /** @type {V} */
    void 0), g === void 0 ? s : g;
  }, i && (r & aa) === 0)
    return y;
  if (p) {
    var T = e.$$legacy;
    return (
      /** @type {() => V} */
      (function(g, b) {
        return arguments.length > 0 ? ((!i || !b || T || m) && p(b ? y() : g), g) : y();
      })
    );
  }
  var d = !1, c = ((r & ua) !== 0 ? pr : On)(() => (d = !1, y()));
  a && h(c);
  var _ = (
    /** @type {Effect} */
    jt
  );
  return (
    /** @type {() => V} */
    (function(g, b) {
      if (arguments.length > 0) {
        const H = b ? h(c) : i && a ? wt(g) : g;
        return w(c, H), d = !0, s !== void 0 && (s = H), g;
      }
      return ha && d || (_.f & Mn) !== 0 ? c.v : h(c);
    })
  );
}
ca();
var Ka = /* @__PURE__ */ kn('<svg class="resize-handle svelte-1stq1b1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><line x1="1" y1="9" x2="9" y2="1" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line><line x1="5" y1="9" x2="9" y2="5" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line></svg>'), on = /* @__PURE__ */ re("<!> <!>", 1), $a = /* @__PURE__ */ re('<div class="placeholder svelte-1stq1b1"></div>');
function es(e, t) {
  qt(t, !1);
  let r = S(t, "height", 8, void 0), n = S(t, "min_height", 8, void 0), i = S(t, "max_height", 8, void 0), a = S(t, "width", 8, void 0), o = S(t, "elem_id", 8, ""), s = S(t, "elem_classes", 24, () => []), u = S(t, "variant", 8, "solid"), l = S(t, "border_mode", 8, "base"), f = S(t, "padding", 8, !0), p = S(t, "type", 8, "normal"), v = S(t, "test_id", 8, void 0), E = S(t, "explicit_call", 8, !1), m = S(t, "container", 8, !0), y = S(t, "visible", 8, !0), T = S(t, "allow_overflow", 8, !0), d = S(t, "overflow_behavior", 8, "auto"), c = S(t, "scale", 8, null), _ = S(t, "min_width", 8, 0), g = S(t, "flex", 12, !1), b = S(t, "resizable", 8, !1), H = S(t, "rtl", 8, !1), P = S(t, "fullscreen", 12, !1), B = S(t, "label", 8, void 0), M = Ze(P()), D = Ze(), J = p() === "fieldset" ? "fieldset" : "div", ne = Ze(0), ie = Ze(0), U = Ze(null);
  function De(K) {
    P() && K.key === "Escape" && P(!1);
  }
  const xe = (K) => {
    if (K !== void 0) {
      if (typeof K == "number")
        return K + "px";
      if (typeof K == "string")
        return K;
    }
  }, Ee = (K) => {
    let Pe = K.clientY;
    const de = ($) => {
      const ee = $.clientY - Pe;
      Pe = $.clientY, pa(D, h(D).style.height = `${h(D).offsetHeight + ee}px`);
    }, Be = () => {
      window.removeEventListener("mousemove", de), window.removeEventListener("mouseup", Be);
    };
    window.addEventListener("mousemove", de), window.addEventListener("mouseup", Be);
  };
  Jr(
    () => (pe(P()), h(M), h(D)),
    () => {
      P() !== h(M) && (w(M, P()), P() ? (w(U, h(D).getBoundingClientRect()), w(ne, h(D).offsetHeight), w(ie, h(D).offsetWidth), window.addEventListener("keydown", De)) : (w(U, null), window.removeEventListener("keydown", De)));
    }
  ), Jr(() => pe(y()), () => {
    y() || g(!1);
  }), da(), Wa();
  var Ve = it(), we = fe(Ve);
  {
    var $e = (K) => {
      var Pe = on(), de = fe(Pe);
      La(de, () => J, !1, (ee, ke) => {
        kr(ee, (A) => w(D, A), () => h(D)), qa(
          ee,
          (A, O) => ({
            "data-testid": v(),
            id: o(),
            class: `block ${A ?? ""}`,
            dir: H() ? "rtl" : "ltr",
            "aria-label": B(),
            style: "",
            [Et]: {
              hidden: y() === "hidden",
              padded: f(),
              flex: g(),
              border_focus: l() === "focus",
              border_contrast: l() === "contrast",
              "hide-container": !E() && !m(),
              fullscreen: P(),
              animating: P() && h(U) !== null,
              "auto-margin": c() === null
            },
            [nt]: O
          }),
          [
            () => (pe(s()), te(() => s()?.join(" ") || "")),
            () => ({
              height: (pe(P()), pe(r()), te(() => P() ? void 0 : xe(r()))),
              "min-height": (pe(P()), pe(n()), te(() => P() ? void 0 : xe(n()))),
              "max-height": (pe(P()), pe(i()), te(() => P() ? void 0 : xe(i()))),
              "--start-top": (h(U), te(() => h(U) ? `${h(U).top}px` : "0px")),
              "--start-left": (h(U), te(() => h(U) ? `${h(U).left}px` : "0px")),
              "--start-width": (h(U), te(() => h(U) ? `${h(U).width}px` : "0px")),
              "--start-height": (h(U), te(() => h(U) ? `${h(U).height}px` : "0px")),
              width: (pe(P()), pe(a()), te(() => P() ? void 0 : typeof a() == "number" ? `calc(min(${a()}px, 100%))` : xe(a()))),
              "border-style": u(),
              overflow: T() ? d() : "hidden",
              "flex-grow": c(),
              "min-width": `calc(min(${_()}px, 100%))`
            })
          ],
          void 0,
          void 0,
          "svelte-1stq1b1"
        );
        var Ie = on(), ae = fe(Ie);
        br(ae, t, "default", {});
        var x = F(ae, 2);
        {
          var I = (A) => {
            var O = Ka();
            Le("mousedown", O, Ee), L(A, O);
          };
          z(x, (A) => {
            b() && A(I);
          });
        }
        L(ke, Ie);
      });
      var Be = F(de, 2);
      {
        var $ = (ee) => {
          var ke = $a();
          let Ie;
          W(() => Ie = ye(ke, "", Ie, {
            height: h(ne) + "px",
            width: h(ie) + "px"
          })), L(ee, ke);
        };
        z(Be, (ee) => {
          P() && ee($);
        });
      }
      L(K, Pe);
    };
    z(we, (K) => {
      (y() === !0 || y() === "hidden") && K($e);
    });
  }
  L(e, Ve), Xt();
}
var ts = /* @__PURE__ */ re('<span class="svelte-vvirtv"> </span>'), rs = /* @__PURE__ */ re("<button><!> <div><!> <!></div></button>");
function ln(e, t) {
  let r = S(t, "label", 3, ""), n = S(t, "show_label", 3, !1), i = S(t, "pending", 3, !1), a = S(t, "size", 3, "small"), o = S(t, "padded", 3, !0), s = S(t, "highlight", 3, !1), u = S(t, "disabled", 3, !1), l = S(t, "hasPopup", 3, !1), f = S(t, "color", 3, "var(--block-label-text-color)"), p = S(t, "transparent", 3, !1), v = S(t, "background", 3, "var(--block-background-fill)"), E = S(t, "border", 3, "transparent"), m = ge(() => s() ? "var(--color-accent)" : f());
  var y = rs();
  let T, d;
  var c = Y(y);
  {
    var _ = (M) => {
      var D = ts(), J = Y(D);
      W(() => ue(J, r())), L(M, D);
    };
    z(c, (M) => {
      n() && M(_);
    });
  }
  var g = F(c, 2);
  let b;
  var H = Y(g);
  Oa(H, () => t.Icon, (M, D) => {
    D(M, {});
  });
  var P = F(H, 2);
  {
    var B = (M) => {
      var D = it(), J = fe(D);
      Ha(J, () => t.children), L(M, D);
    };
    z(P, (M) => {
      t.children && M(B);
    });
  }
  W(() => {
    T = Re(y, 1, "icon-button svelte-vvirtv", null, T, {
      pending: i(),
      padded: o(),
      highlight: s(),
      transparent: p()
    }), y.disabled = u(), at(y, "aria-label", r()), at(y, "aria-haspopup", l()), at(y, "title", r()), d = ye(y, "", d, {
      "--border-color": E(),
      color: !u() && h(m) ? h(m) : "var(--block-label-text-color)",
      "--bg-color": u() ? "auto" : v()
    }), b = Re(g, 1, "svelte-vvirtv", null, b, {
      "x-small": a() === "x-small",
      small: a() === "small",
      large: a() === "large",
      medium: a() === "medium"
    });
  }), Ln("click", y, function(...M) {
    t.onclick?.apply(this, M);
  }), L(e, y);
}
zt(["click"]);
var ns = /* @__PURE__ */ kn('<svg width="100%" height="100%" viewBox="0 0 24 24" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:space="preserve" stroke="currentColor" style="fill-rule:evenodd;clip-rule:evenodd;stroke-linecap:round;stroke-linejoin:round;"><g transform="matrix(1.14096,-0.140958,-0.140958,1.14096,-0.0559523,0.0559523)"><path d="M18,6L6.087,17.913" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></g><path d="M4.364,4.364L19.636,19.636" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></svg>');
function un(e) {
  var t = ns();
  L(e, t);
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
], fn = {
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
    primary: fn[t][r],
    secondary: fn[t][n]
  }
}), {});
function as(e) {
  return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, "default") ? e.default : e;
}
var ir, hn;
function ss() {
  if (hn) return ir;
  hn = 1;
  var e = function(c) {
    return t(c) && !r(c);
  };
  function t(d) {
    return !!d && typeof d == "object";
  }
  function r(d) {
    var c = Object.prototype.toString.call(d);
    return c === "[object RegExp]" || c === "[object Date]" || a(d);
  }
  var n = typeof Symbol == "function" && Symbol.for, i = n ? /* @__PURE__ */ Symbol.for("react.element") : 60103;
  function a(d) {
    return d.$$typeof === i;
  }
  function o(d) {
    return Array.isArray(d) ? [] : {};
  }
  function s(d, c) {
    return c.clone !== !1 && c.isMergeableObject(d) ? y(o(d), d, c) : d;
  }
  function u(d, c, _) {
    return d.concat(c).map(function(g) {
      return s(g, _);
    });
  }
  function l(d, c) {
    if (!c.customMerge)
      return y;
    var _ = c.customMerge(d);
    return typeof _ == "function" ? _ : y;
  }
  function f(d) {
    return Object.getOwnPropertySymbols ? Object.getOwnPropertySymbols(d).filter(function(c) {
      return Object.propertyIsEnumerable.call(d, c);
    }) : [];
  }
  function p(d) {
    return Object.keys(d).concat(f(d));
  }
  function v(d, c) {
    try {
      return c in d;
    } catch {
      return !1;
    }
  }
  function E(d, c) {
    return v(d, c) && !(Object.hasOwnProperty.call(d, c) && Object.propertyIsEnumerable.call(d, c));
  }
  function m(d, c, _) {
    var g = {};
    return _.isMergeableObject(d) && p(d).forEach(function(b) {
      g[b] = s(d[b], _);
    }), p(c).forEach(function(b) {
      E(d, b) || (v(d, b) && _.isMergeableObject(c[b]) ? g[b] = l(b, _)(d[b], c[b], _) : g[b] = s(c[b], _));
    }), g;
  }
  function y(d, c, _) {
    _ = _ || {}, _.arrayMerge = _.arrayMerge || u, _.isMergeableObject = _.isMergeableObject || e, _.cloneUnlessOtherwiseSpecified = s;
    var g = Array.isArray(c), b = Array.isArray(d), H = g === b;
    return H ? g ? _.arrayMerge(d, c, _) : m(d, c, _) : s(c, _);
  }
  y.all = function(c, _) {
    if (!Array.isArray(c))
      throw new Error("first argument should be an array");
    return c.reduce(function(g, b) {
      return y(g, b, _);
    }, {});
  };
  var T = y;
  return ir = T, ir;
}
var os = ss();
const ls = /* @__PURE__ */ as(os);
var yr = function(e, t) {
  return yr = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(r, n) {
    r.__proto__ = n;
  } || function(r, n) {
    for (var i in n) Object.prototype.hasOwnProperty.call(n, i) && (r[i] = n[i]);
  }, yr(e, t);
};
function Zt(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Class extends value " + String(t) + " is not a constructor or null");
  yr(e, t);
  function r() {
    this.constructor = e;
  }
  e.prototype = t === null ? Object.create(t) : (r.prototype = t.prototype, new r());
}
var R = function() {
  return R = Object.assign || function(t) {
    for (var r, n = 1, i = arguments.length; n < i; n++) {
      r = arguments[n];
      for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (t[a] = r[a]);
    }
    return t;
  }, R.apply(this, arguments);
};
function us(e, t) {
  var r = {};
  for (var n in e) Object.prototype.hasOwnProperty.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && typeof Object.getOwnPropertySymbols == "function")
    for (var i = 0, n = Object.getOwnPropertySymbols(e); i < n.length; i++)
      t.indexOf(n[i]) < 0 && Object.prototype.propertyIsEnumerable.call(e, n[i]) && (r[n[i]] = e[n[i]]);
  return r;
}
function ar(e, t, r) {
  if (r || arguments.length === 2) for (var n = 0, i = t.length, a; n < i; n++)
    (a || !(n in t)) && (a || (a = Array.prototype.slice.call(t, 0, n)), a[n] = t[n]);
  return e.concat(a || Array.prototype.slice.call(t));
}
function sr(e, t) {
  var r = t && t.cache ? t.cache : ms, n = t && t.serializer ? t.serializer : ps, i = t && t.strategy ? t.strategy : cs;
  return i(e, {
    cache: r,
    serializer: n
  });
}
function fs(e) {
  return e == null || typeof e == "number" || typeof e == "boolean";
}
function hs(e, t, r, n) {
  var i = fs(n) ? n : r(n), a = t.get(i);
  return typeof a > "u" && (a = e.call(this, n), t.set(i, a)), a;
}
function Xn(e, t, r) {
  var n = Array.prototype.slice.call(arguments, 3), i = r(n), a = t.get(i);
  return typeof a > "u" && (a = e.apply(this, n), t.set(i, a)), a;
}
function qn(e, t, r, n, i) {
  return r.bind(t, e, n, i);
}
function cs(e, t) {
  var r = e.length === 1 ? hs : Xn;
  return qn(e, this, r, t.cache.create(), t.serializer);
}
function ds(e, t) {
  return qn(e, this, Xn, t.cache.create(), t.serializer);
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
}, or = {
  variadic: ds
}, N;
(function(e) {
  e[e.EXPECT_ARGUMENT_CLOSING_BRACE = 1] = "EXPECT_ARGUMENT_CLOSING_BRACE", e[e.EMPTY_ARGUMENT = 2] = "EMPTY_ARGUMENT", e[e.MALFORMED_ARGUMENT = 3] = "MALFORMED_ARGUMENT", e[e.EXPECT_ARGUMENT_TYPE = 4] = "EXPECT_ARGUMENT_TYPE", e[e.INVALID_ARGUMENT_TYPE = 5] = "INVALID_ARGUMENT_TYPE", e[e.EXPECT_ARGUMENT_STYLE = 6] = "EXPECT_ARGUMENT_STYLE", e[e.INVALID_NUMBER_SKELETON = 7] = "INVALID_NUMBER_SKELETON", e[e.INVALID_DATE_TIME_SKELETON = 8] = "INVALID_DATE_TIME_SKELETON", e[e.EXPECT_NUMBER_SKELETON = 9] = "EXPECT_NUMBER_SKELETON", e[e.EXPECT_DATE_TIME_SKELETON = 10] = "EXPECT_DATE_TIME_SKELETON", e[e.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE = 11] = "UNCLOSED_QUOTE_IN_ARGUMENT_STYLE", e[e.EXPECT_SELECT_ARGUMENT_OPTIONS = 12] = "EXPECT_SELECT_ARGUMENT_OPTIONS", e[e.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE = 13] = "EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE = 14] = "INVALID_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR = 15] = "EXPECT_SELECT_ARGUMENT_SELECTOR", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR = 16] = "EXPECT_PLURAL_ARGUMENT_SELECTOR", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT = 17] = "EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT = 18] = "EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT", e[e.INVALID_PLURAL_ARGUMENT_SELECTOR = 19] = "INVALID_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_PLURAL_ARGUMENT_SELECTOR = 20] = "DUPLICATE_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_SELECT_ARGUMENT_SELECTOR = 21] = "DUPLICATE_SELECT_ARGUMENT_SELECTOR", e[e.MISSING_OTHER_CLAUSE = 22] = "MISSING_OTHER_CLAUSE", e[e.INVALID_TAG = 23] = "INVALID_TAG", e[e.INVALID_TAG_NAME = 25] = "INVALID_TAG_NAME", e[e.UNMATCHED_CLOSING_TAG = 26] = "UNMATCHED_CLOSING_TAG", e[e.UNCLOSED_TAG = 27] = "UNCLOSED_TAG";
})(N || (N = {}));
var j;
(function(e) {
  e[e.literal = 0] = "literal", e[e.argument = 1] = "argument", e[e.number = 2] = "number", e[e.date = 3] = "date", e[e.time = 4] = "time", e[e.select = 5] = "select", e[e.plural = 6] = "plural", e[e.pound = 7] = "pound", e[e.tag = 8] = "tag";
})(j || (j = {}));
var ot;
(function(e) {
  e[e.number = 0] = "number", e[e.dateTime = 1] = "dateTime";
})(ot || (ot = {}));
function cn(e) {
  return e.type === j.literal;
}
function gs(e) {
  return e.type === j.argument;
}
function Wn(e) {
  return e.type === j.number;
}
function Zn(e) {
  return e.type === j.date;
}
function Yn(e) {
  return e.type === j.time;
}
function Jn(e) {
  return e.type === j.select;
}
function Qn(e) {
  return e.type === j.plural;
}
function bs(e) {
  return e.type === j.pound;
}
function Kn(e) {
  return e.type === j.tag;
}
function $n(e) {
  return !!(e && typeof e == "object" && e.type === ot.number);
}
function xr(e) {
  return !!(e && typeof e == "object" && e.type === ot.dateTime);
}
var ei = /[ \xA0\u1680\u2000-\u200A\u202F\u205F\u3000]/, _s = /(?:[Eec]{1,6}|G{1,5}|[Qq]{1,5}|(?:[yYur]+|U{1,5})|[ML]{1,5}|d{1,2}|D{1,3}|F{1}|[abB]{1,5}|[hkHK]{1,2}|w{1,2}|W{1}|m{1,2}|s{1,2}|[zZOvVxX]{1,4})(?=([^']*'[^']*')*[^']*$)/g;
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
    for (var s = o[0], u = o.slice(1), l = 0, f = u; l < f.length; l++) {
      var p = f[l];
      if (p.length === 0)
        throw new Error("Invalid number skeleton");
    }
    r.push({ stem: s, options: u });
  }
  return r;
}
function ws(e) {
  return e.replace(/^(.*?)-/, "");
}
var dn = /^\.(?:(0+)(\*)?|(#+)|(0+)(#+))$/g, ti = /^(@+)?(\+|#+)?[rs]?$/g, Ts = /(\*)(0+)|(#+)(0+)|(0+)/g, ri = /^(0+)$/;
function pn(e) {
  var t = {};
  return e[e.length - 1] === "r" ? t.roundingPriority = "morePrecision" : e[e.length - 1] === "s" && (t.roundingPriority = "lessPrecision"), e.replace(ti, function(r, n, i) {
    return typeof i != "string" ? (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length) : i === "+" ? t.minimumSignificantDigits = n.length : n[0] === "#" ? t.maximumSignificantDigits = n.length : (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length + (typeof i == "string" ? i.length : 0)), "";
  }), t;
}
function ni(e) {
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
    if (r === "+!" ? (t.signDisplay = "always", e = e.slice(2)) : r === "+?" && (t.signDisplay = "exceptZero", e = e.slice(2)), !ri.test(e))
      throw new Error("Malformed concise eng/scientific notation");
    t.minimumIntegerDigits = e.length;
  }
  return t;
}
function vn(e) {
  var t = {}, r = ni(e);
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
        t = R(R(R({}, t), { notation: "scientific" }), i.options.reduce(function(u, l) {
          return R(R({}, u), vn(l));
        }, {}));
        continue;
      case "engineering":
        t = R(R(R({}, t), { notation: "engineering" }), i.options.reduce(function(u, l) {
          return R(R({}, u), vn(l));
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
        i.options[0].replace(Ts, function(u, l, f, p, v, E) {
          if (l)
            t.minimumIntegerDigits = f.length;
          else {
            if (p && v)
              throw new Error("We currently do not support maximum integer digits");
            if (E)
              throw new Error("We currently do not support exact integer digits");
          }
          return "";
        });
        continue;
    }
    if (ri.test(i.stem)) {
      t.minimumIntegerDigits = i.stem.length;
      continue;
    }
    if (dn.test(i.stem)) {
      if (i.options.length > 1)
        throw new RangeError("Fraction-precision stems only accept a single optional option");
      i.stem.replace(dn, function(u, l, f, p, v, E) {
        return f === "*" ? t.minimumFractionDigits = l.length : p && p[0] === "#" ? t.maximumFractionDigits = p.length : v && E ? (t.minimumFractionDigits = v.length, t.maximumFractionDigits = v.length + E.length) : (t.minimumFractionDigits = l.length, t.maximumFractionDigits = l.length), "";
      });
      var a = i.options[0];
      a === "w" ? t = R(R({}, t), { trailingZeroDisplay: "stripIfInteger" }) : a && (t = R(R({}, t), pn(a)));
      continue;
    }
    if (ti.test(i.stem)) {
      t = R(R({}, t), pn(i.stem));
      continue;
    }
    var o = ni(i.stem);
    o && (t = R(R({}, t), o));
    var s = Ss(i.stem);
    s && (t = R(R({}, t), s));
  }
  return t;
}
var Nt = {
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
  var i = Nt[n || ""] || Nt[r || ""] || Nt["".concat(r, "-001")] || Nt["001"];
  return i[0];
}
var lr, Bs = new RegExp("^".concat(ei.source, "*")), Is = new RegExp("".concat(ei.source, "*$"));
function C(e, t) {
  return { start: e, end: t };
}
var Os = !!String.prototype.startsWith && "_a".startsWith("a", 1), Ms = !!String.fromCodePoint, Ns = !!Object.fromEntries, Ls = !!String.prototype.codePointAt, Cs = !!String.prototype.trimStart, Rs = !!String.prototype.trimEnd, Ds = !!Number.isSafeInteger, ks = Ds ? Number.isSafeInteger : function(e) {
  return typeof e == "number" && isFinite(e) && Math.floor(e) === e && Math.abs(e) <= 9007199254740991;
}, Er = !0;
try {
  var Us = ai("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  Er = ((lr = Us.exec("a")) === null || lr === void 0 ? void 0 : lr[0]) === "a";
} catch {
  Er = !1;
}
var mn = Os ? (
  // Native
  function(t, r, n) {
    return t.startsWith(r, n);
  }
) : (
  // For IE11
  function(t, r, n) {
    return t.slice(n, n + r.length) === r;
  }
), wr = Ms ? String.fromCodePoint : (
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
), gn = (
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
), ii = Ls ? (
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
    return t.replace(Bs, "");
  }
), Gs = Rs ? (
  // Native
  function(t) {
    return t.trimEnd();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(Is, "");
  }
);
function ai(e, t) {
  return new RegExp(e, t);
}
var Tr;
if (Er) {
  var bn = ai("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  Tr = function(t, r) {
    var n;
    bn.lastIndex = r;
    var i = bn.exec(t);
    return (n = i[1]) !== null && n !== void 0 ? n : "";
  };
} else
  Tr = function(t, r) {
    for (var n = []; ; ) {
      var i = ii(t, r);
      if (i === void 0 || si(i) || Xs(i))
        break;
      n.push(i), r += i >= 65536 ? 2 : 1;
    }
    return wr.apply(void 0, n);
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
              type: j.pound,
              location: C(s, this.clonePosition())
            });
          } else if (a === 60 && !this.ignoreTag && this.peek() === 47) {
            if (n)
              break;
            return this.error(N.UNMATCHED_CLOSING_TAG, C(this.clonePosition(), this.clonePosition()));
          } else if (a === 60 && !this.ignoreTag && Sr(this.peek() || 0)) {
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
            type: j.literal,
            value: "<".concat(i, "/>"),
            location: C(n, this.clonePosition())
          },
          err: null
        };
      if (this.bumpIf(">")) {
        var a = this.parseMessage(t + 1, r, !0);
        if (a.err)
          return a;
        var o = a.val, s = this.clonePosition();
        if (this.bumpIf("</")) {
          if (this.isEOF() || !Sr(this.char()))
            return this.error(N.INVALID_TAG, C(s, this.clonePosition()));
          var u = this.clonePosition(), l = this.parseTagName();
          return i !== l ? this.error(N.UNMATCHED_CLOSING_TAG, C(u, this.clonePosition())) : (this.bumpSpace(), this.bumpIf(">") ? {
            val: {
              type: j.tag,
              value: i,
              children: o,
              location: C(n, this.clonePosition())
            },
            err: null
          } : this.error(N.INVALID_TAG, C(s, this.clonePosition())));
        } else
          return this.error(N.UNCLOSED_TAG, C(n, this.clonePosition()));
      } else
        return this.error(N.INVALID_TAG, C(n, this.clonePosition()));
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
      var u = C(n, this.clonePosition());
      return {
        val: { type: j.literal, value: i, location: u },
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
      return wr.apply(void 0, r);
    }, e.prototype.tryParseUnquoted = function(t, r) {
      if (this.isEOF())
        return null;
      var n = this.char();
      return n === 60 || n === 123 || n === 35 && (r === "plural" || r === "selectordinal") || n === 125 && t > 0 ? null : (this.bump(), wr(n));
    }, e.prototype.parseArgument = function(t, r) {
      var n = this.clonePosition();
      if (this.bump(), this.bumpSpace(), this.isEOF())
        return this.error(N.EXPECT_ARGUMENT_CLOSING_BRACE, C(n, this.clonePosition()));
      if (this.char() === 125)
        return this.bump(), this.error(N.EMPTY_ARGUMENT, C(n, this.clonePosition()));
      var i = this.parseIdentifierIfPossible().value;
      if (!i)
        return this.error(N.MALFORMED_ARGUMENT, C(n, this.clonePosition()));
      if (this.bumpSpace(), this.isEOF())
        return this.error(N.EXPECT_ARGUMENT_CLOSING_BRACE, C(n, this.clonePosition()));
      switch (this.char()) {
        // Simple argument: `{name}`
        case 125:
          return this.bump(), {
            val: {
              type: j.argument,
              // value does not include the opening and closing braces.
              value: i,
              location: C(n, this.clonePosition())
            },
            err: null
          };
        // Argument with options: `{name, format, ...}`
        case 44:
          return this.bump(), this.bumpSpace(), this.isEOF() ? this.error(N.EXPECT_ARGUMENT_CLOSING_BRACE, C(n, this.clonePosition())) : this.parseArgumentOptions(t, r, i, n);
        default:
          return this.error(N.MALFORMED_ARGUMENT, C(n, this.clonePosition()));
      }
    }, e.prototype.parseIdentifierIfPossible = function() {
      var t = this.clonePosition(), r = this.offset(), n = Tr(this.message, r), i = r + n.length;
      this.bumpTo(i);
      var a = this.clonePosition(), o = C(t, a);
      return { value: n, location: o };
    }, e.prototype.parseArgumentOptions = function(t, r, n, i) {
      var a, o = this.clonePosition(), s = this.parseIdentifierIfPossible().value, u = this.clonePosition();
      switch (s) {
        case "":
          return this.error(N.EXPECT_ARGUMENT_TYPE, C(o, u));
        case "number":
        case "date":
        case "time": {
          this.bumpSpace();
          var l = null;
          if (this.bumpIf(",")) {
            this.bumpSpace();
            var f = this.clonePosition(), p = this.parseSimpleArgStyleIfPossible();
            if (p.err)
              return p;
            var v = Gs(p.val);
            if (v.length === 0)
              return this.error(N.EXPECT_ARGUMENT_STYLE, C(this.clonePosition(), this.clonePosition()));
            var E = C(f, this.clonePosition());
            l = { style: v, styleLocation: E };
          }
          var m = this.tryParseArgumentClose(i);
          if (m.err)
            return m;
          var y = C(i, this.clonePosition());
          if (l && mn(l?.style, "::", 0)) {
            var T = Fs(l.style.slice(2));
            if (s === "number") {
              var p = this.parseNumberSkeletonFromString(T, l.styleLocation);
              return p.err ? p : {
                val: { type: j.number, value: n, location: y, style: p.val },
                err: null
              };
            } else {
              if (T.length === 0)
                return this.error(N.EXPECT_DATE_TIME_SKELETON, y);
              var d = T;
              this.locale && (d = Hs(T, this.locale));
              var v = {
                type: ot.dateTime,
                pattern: d,
                location: l.styleLocation,
                parsedOptions: this.shouldParseSkeletons ? ys(d) : {}
              }, c = s === "date" ? j.date : j.time;
              return {
                val: { type: c, value: n, location: y, style: v },
                err: null
              };
            }
          }
          return {
            val: {
              type: s === "number" ? j.number : s === "date" ? j.date : j.time,
              value: n,
              location: y,
              style: (a = l?.style) !== null && a !== void 0 ? a : null
            },
            err: null
          };
        }
        case "plural":
        case "selectordinal":
        case "select": {
          var _ = this.clonePosition();
          if (this.bumpSpace(), !this.bumpIf(","))
            return this.error(N.EXPECT_SELECT_ARGUMENT_OPTIONS, C(_, R({}, _)));
          this.bumpSpace();
          var g = this.parseIdentifierIfPossible(), b = 0;
          if (s !== "select" && g.value === "offset") {
            if (!this.bumpIf(":"))
              return this.error(N.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, C(this.clonePosition(), this.clonePosition()));
            this.bumpSpace();
            var p = this.tryParseDecimalInteger(N.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, N.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE);
            if (p.err)
              return p;
            this.bumpSpace(), g = this.parseIdentifierIfPossible(), b = p.val;
          }
          var H = this.tryParsePluralOrSelectOptions(t, s, r, g);
          if (H.err)
            return H;
          var m = this.tryParseArgumentClose(i);
          if (m.err)
            return m;
          var P = C(i, this.clonePosition());
          return s === "select" ? {
            val: {
              type: j.select,
              value: n,
              options: gn(H.val),
              location: P
            },
            err: null
          } : {
            val: {
              type: j.plural,
              value: n,
              options: gn(H.val),
              offset: b,
              pluralType: s === "plural" ? "cardinal" : "ordinal",
              location: P
            },
            err: null
          };
        }
        default:
          return this.error(N.INVALID_ARGUMENT_TYPE, C(o, u));
      }
    }, e.prototype.tryParseArgumentClose = function(t) {
      return this.isEOF() || this.char() !== 125 ? this.error(N.EXPECT_ARGUMENT_CLOSING_BRACE, C(t, this.clonePosition())) : (this.bump(), { val: !0, err: null });
    }, e.prototype.parseSimpleArgStyleIfPossible = function() {
      for (var t = 0, r = this.clonePosition(); !this.isEOF(); ) {
        var n = this.char();
        switch (n) {
          case 39: {
            this.bump();
            var i = this.clonePosition();
            if (!this.bumpUntil("'"))
              return this.error(N.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE, C(i, this.clonePosition()));
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
        return this.error(N.INVALID_NUMBER_SKELETON, r);
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
      for (var a, o = !1, s = [], u = /* @__PURE__ */ new Set(), l = i.value, f = i.location; ; ) {
        if (l.length === 0) {
          var p = this.clonePosition();
          if (r !== "select" && this.bumpIf("=")) {
            var v = this.tryParseDecimalInteger(N.EXPECT_PLURAL_ARGUMENT_SELECTOR, N.INVALID_PLURAL_ARGUMENT_SELECTOR);
            if (v.err)
              return v;
            f = C(p, this.clonePosition()), l = this.message.slice(p.offset, this.offset());
          } else
            break;
        }
        if (u.has(l))
          return this.error(r === "select" ? N.DUPLICATE_SELECT_ARGUMENT_SELECTOR : N.DUPLICATE_PLURAL_ARGUMENT_SELECTOR, f);
        l === "other" && (o = !0), this.bumpSpace();
        var E = this.clonePosition();
        if (!this.bumpIf("{"))
          return this.error(r === "select" ? N.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT : N.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT, C(this.clonePosition(), this.clonePosition()));
        var m = this.parseMessage(t + 1, r, n);
        if (m.err)
          return m;
        var y = this.tryParseArgumentClose(E);
        if (y.err)
          return y;
        s.push([
          l,
          {
            value: m.val,
            location: C(E, this.clonePosition())
          }
        ]), u.add(l), this.bumpSpace(), a = this.parseIdentifierIfPossible(), l = a.value, f = a.location;
      }
      return s.length === 0 ? this.error(r === "select" ? N.EXPECT_SELECT_ARGUMENT_SELECTOR : N.EXPECT_PLURAL_ARGUMENT_SELECTOR, C(this.clonePosition(), this.clonePosition())) : this.requiresOtherClause && !o ? this.error(N.MISSING_OTHER_CLAUSE, C(this.clonePosition(), this.clonePosition())) : { val: s, err: null };
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
      var u = C(i, this.clonePosition());
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
      var r = ii(this.message, t);
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
      if (mn(this.message, t, this.offset())) {
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
      for (; !this.isEOF() && si(this.char()); )
        this.bump();
    }, e.prototype.peek = function() {
      if (this.isEOF())
        return null;
      var t = this.char(), r = this.offset(), n = this.message.charCodeAt(r + (t >= 65536 ? 2 : 1));
      return n ?? null;
    }, e;
  })()
);
function Sr(e) {
  return e >= 97 && e <= 122 || e >= 65 && e <= 90;
}
function Vs(e) {
  return Sr(e) || e === 47;
}
function zs(e) {
  return e === 45 || e === 46 || e >= 48 && e <= 57 || e === 95 || e >= 97 && e <= 122 || e >= 65 && e <= 90 || e == 183 || e >= 192 && e <= 214 || e >= 216 && e <= 246 || e >= 248 && e <= 893 || e >= 895 && e <= 8191 || e >= 8204 && e <= 8205 || e >= 8255 && e <= 8256 || e >= 8304 && e <= 8591 || e >= 11264 && e <= 12271 || e >= 12289 && e <= 55295 || e >= 63744 && e <= 64975 || e >= 65008 && e <= 65533 || e >= 65536 && e <= 983039;
}
function si(e) {
  return e >= 9 && e <= 13 || e === 32 || e === 133 || e >= 8206 && e <= 8207 || e === 8232 || e === 8233;
}
function Xs(e) {
  return e >= 33 && e <= 35 || e === 36 || e >= 37 && e <= 39 || e === 40 || e === 41 || e === 42 || e === 43 || e === 44 || e === 45 || e >= 46 && e <= 47 || e >= 58 && e <= 59 || e >= 60 && e <= 62 || e >= 63 && e <= 64 || e === 91 || e === 92 || e === 93 || e === 94 || e === 96 || e === 123 || e === 124 || e === 125 || e === 126 || e === 161 || e >= 162 && e <= 165 || e === 166 || e === 167 || e === 169 || e === 171 || e === 172 || e === 174 || e === 176 || e === 177 || e === 182 || e === 187 || e === 191 || e === 215 || e === 247 || e >= 8208 && e <= 8213 || e >= 8214 && e <= 8215 || e === 8216 || e === 8217 || e === 8218 || e >= 8219 && e <= 8220 || e === 8221 || e === 8222 || e === 8223 || e >= 8224 && e <= 8231 || e >= 8240 && e <= 8248 || e === 8249 || e === 8250 || e >= 8251 && e <= 8254 || e >= 8257 && e <= 8259 || e === 8260 || e === 8261 || e === 8262 || e >= 8263 && e <= 8273 || e === 8274 || e === 8275 || e >= 8277 && e <= 8286 || e >= 8592 && e <= 8596 || e >= 8597 && e <= 8601 || e >= 8602 && e <= 8603 || e >= 8604 && e <= 8607 || e === 8608 || e >= 8609 && e <= 8610 || e === 8611 || e >= 8612 && e <= 8613 || e === 8614 || e >= 8615 && e <= 8621 || e === 8622 || e >= 8623 && e <= 8653 || e >= 8654 && e <= 8655 || e >= 8656 && e <= 8657 || e === 8658 || e === 8659 || e === 8660 || e >= 8661 && e <= 8691 || e >= 8692 && e <= 8959 || e >= 8960 && e <= 8967 || e === 8968 || e === 8969 || e === 8970 || e === 8971 || e >= 8972 && e <= 8991 || e >= 8992 && e <= 8993 || e >= 8994 && e <= 9e3 || e === 9001 || e === 9002 || e >= 9003 && e <= 9083 || e === 9084 || e >= 9085 && e <= 9114 || e >= 9115 && e <= 9139 || e >= 9140 && e <= 9179 || e >= 9180 && e <= 9185 || e >= 9186 && e <= 9254 || e >= 9255 && e <= 9279 || e >= 9280 && e <= 9290 || e >= 9291 && e <= 9311 || e >= 9472 && e <= 9654 || e === 9655 || e >= 9656 && e <= 9664 || e === 9665 || e >= 9666 && e <= 9719 || e >= 9720 && e <= 9727 || e >= 9728 && e <= 9838 || e === 9839 || e >= 9840 && e <= 10087 || e === 10088 || e === 10089 || e === 10090 || e === 10091 || e === 10092 || e === 10093 || e === 10094 || e === 10095 || e === 10096 || e === 10097 || e === 10098 || e === 10099 || e === 10100 || e === 10101 || e >= 10132 && e <= 10175 || e >= 10176 && e <= 10180 || e === 10181 || e === 10182 || e >= 10183 && e <= 10213 || e === 10214 || e === 10215 || e === 10216 || e === 10217 || e === 10218 || e === 10219 || e === 10220 || e === 10221 || e === 10222 || e === 10223 || e >= 10224 && e <= 10239 || e >= 10240 && e <= 10495 || e >= 10496 && e <= 10626 || e === 10627 || e === 10628 || e === 10629 || e === 10630 || e === 10631 || e === 10632 || e === 10633 || e === 10634 || e === 10635 || e === 10636 || e === 10637 || e === 10638 || e === 10639 || e === 10640 || e === 10641 || e === 10642 || e === 10643 || e === 10644 || e === 10645 || e === 10646 || e === 10647 || e === 10648 || e >= 10649 && e <= 10711 || e === 10712 || e === 10713 || e === 10714 || e === 10715 || e >= 10716 && e <= 10747 || e === 10748 || e === 10749 || e >= 10750 && e <= 11007 || e >= 11008 && e <= 11055 || e >= 11056 && e <= 11076 || e >= 11077 && e <= 11078 || e >= 11079 && e <= 11084 || e >= 11085 && e <= 11123 || e >= 11124 && e <= 11125 || e >= 11126 && e <= 11157 || e === 11158 || e >= 11159 && e <= 11263 || e >= 11776 && e <= 11777 || e === 11778 || e === 11779 || e === 11780 || e === 11781 || e >= 11782 && e <= 11784 || e === 11785 || e === 11786 || e === 11787 || e === 11788 || e === 11789 || e >= 11790 && e <= 11798 || e === 11799 || e >= 11800 && e <= 11801 || e === 11802 || e === 11803 || e === 11804 || e === 11805 || e >= 11806 && e <= 11807 || e === 11808 || e === 11809 || e === 11810 || e === 11811 || e === 11812 || e === 11813 || e === 11814 || e === 11815 || e === 11816 || e === 11817 || e >= 11818 && e <= 11822 || e === 11823 || e >= 11824 && e <= 11833 || e >= 11834 && e <= 11835 || e >= 11836 && e <= 11839 || e === 11840 || e === 11841 || e === 11842 || e >= 11843 && e <= 11855 || e >= 11856 && e <= 11857 || e === 11858 || e >= 11859 && e <= 11903 || e >= 12289 && e <= 12291 || e === 12296 || e === 12297 || e === 12298 || e === 12299 || e === 12300 || e === 12301 || e === 12302 || e === 12303 || e === 12304 || e === 12305 || e >= 12306 && e <= 12307 || e === 12308 || e === 12309 || e === 12310 || e === 12311 || e === 12312 || e === 12313 || e === 12314 || e === 12315 || e === 12316 || e === 12317 || e >= 12318 && e <= 12319 || e === 12320 || e === 12336 || e === 64830 || e === 64831 || e >= 65093 && e <= 65094;
}
function Ar(e) {
  e.forEach(function(t) {
    if (delete t.location, Jn(t) || Qn(t))
      for (var r in t.options)
        delete t.options[r].location, Ar(t.options[r].value);
    else Wn(t) && $n(t.style) || (Zn(t) || Yn(t)) && xr(t.style) ? delete t.style.location : Kn(t) && Ar(t.children);
  });
}
function qs(e, t) {
  t === void 0 && (t = {}), t = R({ shouldParseSkeletons: !0, requiresOtherClause: !0 }, t);
  var r = new js(e, t).parse();
  if (r.err) {
    var n = SyntaxError(N[r.err.kind]);
    throw n.location = r.err.location, n.originalMessage = r.err.message, n;
  }
  return t?.captureLocation || Ar(r.val), r.val;
}
var lt;
(function(e) {
  e.MISSING_VALUE = "MISSING_VALUE", e.INVALID_VALUE = "INVALID_VALUE", e.MISSING_INTL_API = "MISSING_INTL_API";
})(lt || (lt = {}));
var Yt = (
  /** @class */
  (function(e) {
    Zt(t, e);
    function t(r, n, i) {
      var a = e.call(this, r) || this;
      return a.code = n, a.originalMessage = i, a;
    }
    return t.prototype.toString = function() {
      return "[formatjs Error: ".concat(this.code, "] ").concat(this.message);
    }, t;
  })(Error)
), _n = (
  /** @class */
  (function(e) {
    Zt(t, e);
    function t(r, n, i, a) {
      return e.call(this, 'Invalid values for "'.concat(r, '": "').concat(n, '". Options are "').concat(Object.keys(i).join('", "'), '"'), lt.INVALID_VALUE, a) || this;
    }
    return t;
  })(Yt)
), Ws = (
  /** @class */
  (function(e) {
    Zt(t, e);
    function t(r, n, i) {
      return e.call(this, 'Value for "'.concat(r, '" must be of type ').concat(n), lt.INVALID_VALUE, i) || this;
    }
    return t;
  })(Yt)
), Zs = (
  /** @class */
  (function(e) {
    Zt(t, e);
    function t(r, n) {
      return e.call(this, 'The intl string context variable "'.concat(r, '" was not provided to the string "').concat(n, '"'), lt.MISSING_VALUE, n) || this;
    }
    return t;
  })(Yt)
), he;
(function(e) {
  e[e.literal = 0] = "literal", e[e.object = 1] = "object";
})(he || (he = {}));
function Ys(e) {
  return e.length < 2 ? e : e.reduce(function(t, r) {
    var n = t[t.length - 1];
    return !n || n.type !== he.literal || r.type !== he.literal ? t.push(r) : n.value += r.value, t;
  }, []);
}
function Js(e) {
  return typeof e == "function";
}
function Rt(e, t, r, n, i, a, o) {
  if (e.length === 1 && cn(e[0]))
    return [
      {
        type: he.literal,
        value: e[0].value
      }
    ];
  for (var s = [], u = 0, l = e; u < l.length; u++) {
    var f = l[u];
    if (cn(f)) {
      s.push({
        type: he.literal,
        value: f.value
      });
      continue;
    }
    if (bs(f)) {
      typeof a == "number" && s.push({
        type: he.literal,
        value: r.getNumberFormat(t).format(a)
      });
      continue;
    }
    var p = f.value;
    if (!(i && p in i))
      throw new Zs(p, o);
    var v = i[p];
    if (gs(f)) {
      (!v || typeof v == "string" || typeof v == "number") && (v = typeof v == "string" || typeof v == "number" ? String(v) : ""), s.push({
        type: typeof v == "string" ? he.literal : he.object,
        value: v
      });
      continue;
    }
    if (Zn(f)) {
      var E = typeof f.style == "string" ? n.date[f.style] : xr(f.style) ? f.style.parsedOptions : void 0;
      s.push({
        type: he.literal,
        value: r.getDateTimeFormat(t, E).format(v)
      });
      continue;
    }
    if (Yn(f)) {
      var E = typeof f.style == "string" ? n.time[f.style] : xr(f.style) ? f.style.parsedOptions : n.time.medium;
      s.push({
        type: he.literal,
        value: r.getDateTimeFormat(t, E).format(v)
      });
      continue;
    }
    if (Wn(f)) {
      var E = typeof f.style == "string" ? n.number[f.style] : $n(f.style) ? f.style.parsedOptions : void 0;
      E && E.scale && (v = v * (E.scale || 1)), s.push({
        type: he.literal,
        value: r.getNumberFormat(t, E).format(v)
      });
      continue;
    }
    if (Kn(f)) {
      var m = f.children, y = f.value, T = i[y];
      if (!Js(T))
        throw new Ws(y, "function", o);
      var d = Rt(m, t, r, n, i, a), c = T(d.map(function(b) {
        return b.value;
      }));
      Array.isArray(c) || (c = [c]), s.push.apply(s, c.map(function(b) {
        return {
          type: typeof b == "string" ? he.literal : he.object,
          value: b
        };
      }));
    }
    if (Jn(f)) {
      var _ = f.options[v] || f.options.other;
      if (!_)
        throw new _n(f.value, v, Object.keys(f.options), o);
      s.push.apply(s, Rt(_.value, t, r, n, i));
      continue;
    }
    if (Qn(f)) {
      var _ = f.options["=".concat(v)];
      if (!_) {
        if (!Intl.PluralRules)
          throw new Yt(`Intl.PluralRules is not available in this environment.
Try polyfilling it using "@formatjs/intl-pluralrules"
`, lt.MISSING_INTL_API, o);
        var g = r.getPluralRules(t, { type: f.pluralType }).select(v - (f.offset || 0));
        _ = f.options[g] || f.options.other;
      }
      if (!_)
        throw new _n(f.value, v, Object.keys(f.options), o);
      s.push.apply(s, Rt(_.value, t, r, n, i, v - (f.offset || 0)));
      continue;
    }
  }
  return Ys(s);
}
function Qs(e, t) {
  return t ? R(R(R({}, e || {}), t || {}), Object.keys(e).reduce(function(r, n) {
    return r[n] = R(R({}, e[n]), t[n] || {}), r;
  }, {})) : e;
}
function Ks(e, t) {
  return t ? Object.keys(e).reduce(function(r, n) {
    return r[n] = Qs(e[n], t[n]), r;
  }, R({}, e)) : e;
}
function ur(e) {
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
    getNumberFormat: sr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.NumberFormat).bind.apply(t, ar([void 0], r, !1)))();
    }, {
      cache: ur(e.number),
      strategy: or.variadic
    }),
    getDateTimeFormat: sr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.DateTimeFormat).bind.apply(t, ar([void 0], r, !1)))();
    }, {
      cache: ur(e.dateTime),
      strategy: or.variadic
    }),
    getPluralRules: sr(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.PluralRules).bind.apply(t, ar([void 0], r, !1)))();
    }, {
      cache: ur(e.pluralRules),
      strategy: or.variadic
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
        var f = l.reduce(function(p, v) {
          return !p.length || v.type !== he.literal || typeof p[p.length - 1] != "string" ? p.push(v.value) : p[p.length - 1] += v.value, p;
        }, []);
        return f.length <= 1 ? f[0] || "" : f;
      }, this.formatToParts = function(u) {
        return Rt(a.ast, a.locales, a.formatters, a.formats, u, void 0, a.message);
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
        this.ast = e.__parse(t, R(R({}, s), { locale: this.resolvedLocale }));
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
const Ge = {}, ro = (e, t, r) => r && (t in Ge || (Ge[t] = {}), e in Ge[t] || (Ge[t][e] = r), r), oi = (e, t) => {
  if (t == null)
    return;
  if (t in Ge && e in Ge[t])
    return Ge[t][e];
  const r = Jt(t);
  for (let n = 0; n < r.length; n++) {
    const i = r[n], a = io(i, e);
    if (a)
      return ro(e, t, a);
  }
};
let Ur;
const Ht = At({});
function no(e) {
  return Ur[e] || null;
}
function li(e) {
  return e in Ur;
}
function io(e, t) {
  if (!li(e))
    return null;
  const r = no(e);
  return to(r, t);
}
function ao(e) {
  if (e == null)
    return;
  const t = Jt(e);
  for (let r = 0; r < t.length; r++) {
    const n = t[r];
    if (li(n))
      return n;
  }
}
function so(e, ...t) {
  delete Ge[e], Ht.update((r) => (r[e] = ls.all([r[e] || {}, ...t]), r));
}
ft(
  [Ht],
  ([e]) => Object.keys(e)
);
Ht.subscribe((e) => Ur = e);
const Dt = {};
function oo(e, t) {
  Dt[e].delete(t), Dt[e].size === 0 && delete Dt[e];
}
function ui(e) {
  return Dt[e];
}
function lo(e) {
  return Jt(e).map((t) => {
    const r = ui(t);
    return [t, r ? [...r] : []];
  }).filter(([, t]) => t.length > 0);
}
function Hr(e) {
  return e == null ? !1 : Jt(e).some(
    (t) => {
      var r;
      return (r = ui(t)) == null ? void 0 : r.size;
    }
  );
}
function uo(e, t) {
  return Promise.all(
    t.map((n) => (oo(e, n), n().then((i) => i.default || i)))
  ).then((n) => so(e, ...n));
}
const yt = {};
function fi(e) {
  if (!Hr(e))
    return e in yt ? yt[e] : Promise.resolve();
  const t = lo(e);
  return yt[e] = Promise.all(
    t.map(
      ([r, n]) => uo(r, n)
    )
  ).then(() => {
    if (Hr(e))
      return fi(e);
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
}, ho = {
  fallbackLocale: null,
  loadingDelay: 200,
  formats: fo,
  warnOnMissingMessages: !0,
  handleMissingMessage: void 0,
  ignoreTag: !0
}, co = ho;
function ut() {
  return co;
}
const fr = At(!1);
var po = Object.defineProperty, vo = Object.defineProperties, mo = Object.getOwnPropertyDescriptors, yn = Object.getOwnPropertySymbols, go = Object.prototype.hasOwnProperty, bo = Object.prototype.propertyIsEnumerable, xn = (e, t, r) => t in e ? po(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, _o = (e, t) => {
  for (var r in t || (t = {}))
    go.call(t, r) && xn(e, r, t[r]);
  if (yn)
    for (var r of yn(t))
      bo.call(t, r) && xn(e, r, t[r]);
  return e;
}, yo = (e, t) => vo(e, mo(t));
let Pr;
const Ft = At(null);
function En(e) {
  return e.split("-").map((t, r, n) => n.slice(0, r + 1).join("-")).reverse();
}
function Jt(e, t = ut().fallbackLocale) {
  const r = En(e);
  return t ? [.../* @__PURE__ */ new Set([...r, ...En(t)])] : r;
}
function Ke() {
  return Pr ?? void 0;
}
Ft.subscribe((e) => {
  Pr = e ?? void 0, typeof window < "u" && e != null && document.documentElement.setAttribute("lang", e);
});
const xo = (e) => {
  if (e && ao(e) && Hr(e)) {
    const { loadingDelay: t } = ut();
    let r;
    return typeof window < "u" && Ke() != null && t ? r = window.setTimeout(
      () => fr.set(!0),
      t
    ) : fr.set(!0), fi(e).then(() => {
      Ft.set(e);
    }).finally(() => {
      clearTimeout(r), fr.set(!1);
    });
  }
  return Ft.set(e);
}, ht = yo(_o({}, Ft), {
  set: xo
}), Qt = (e) => {
  const t = /* @__PURE__ */ Object.create(null);
  return (n) => {
    const i = JSON.stringify(n);
    return i in t ? t[i] : t[i] = e(n);
  };
};
var Eo = Object.defineProperty, Gt = Object.getOwnPropertySymbols, hi = Object.prototype.hasOwnProperty, ci = Object.prototype.propertyIsEnumerable, wn = (e, t, r) => t in e ? Eo(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, Fr = (e, t) => {
  for (var r in t || (t = {}))
    hi.call(t, r) && wn(e, r, t[r]);
  if (Gt)
    for (var r of Gt(t))
      ci.call(t, r) && wn(e, r, t[r]);
  return e;
}, ct = (e, t) => {
  var r = {};
  for (var n in e)
    hi.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && Gt)
    for (var n of Gt(e))
      t.indexOf(n) < 0 && ci.call(e, n) && (r[n] = e[n]);
  return r;
};
const Tt = (e, t) => {
  const { formats: r } = ut();
  if (e in r && t in r[e])
    return r[e][t];
  throw new Error(`[svelte-i18n] Unknown "${t}" ${e} format.`);
}, wo = Qt(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = ct(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format numbers');
    return n && (i = Tt("number", n)), new Intl.NumberFormat(r, i);
  }
), To = Qt(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = ct(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format dates');
    return n ? i = Tt("date", n) : Object.keys(i).length === 0 && (i = Tt("date", "short")), new Intl.DateTimeFormat(r, i);
  }
), So = Qt(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = ct(t, ["locale", "format"]);
    if (r == null)
      throw new Error(
        '[svelte-i18n] A "locale" must be set to format time values'
      );
    return n ? i = Tt("time", n) : Object.keys(i).length === 0 && (i = Tt("time", "short")), new Intl.DateTimeFormat(r, i);
  }
), Ao = (e = {}) => {
  var t = e, {
    locale: r = Ke()
  } = t, n = ct(t, [
    "locale"
  ]);
  return wo(Fr({ locale: r }, n));
}, Ho = (e = {}) => {
  var t = e, {
    locale: r = Ke()
  } = t, n = ct(t, [
    "locale"
  ]);
  return To(Fr({ locale: r }, n));
}, Po = (e = {}) => {
  var t = e, {
    locale: r = Ke()
  } = t, n = ct(t, [
    "locale"
  ]);
  return So(Fr({ locale: r }, n));
}, Bo = Qt(
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  (e, t = Ke()) => new eo(e, t, ut().formats, {
    ignoreTag: ut().ignoreTag
  })
), Io = (e, t = {}) => {
  var r, n, i, a;
  let o = t;
  typeof e == "object" && (o = e, e = o.id);
  const {
    values: s,
    locale: u = Ke(),
    default: l
  } = o;
  if (u == null)
    throw new Error(
      "[svelte-i18n] Cannot format a message without first setting the initial locale."
    );
  let f = oi(e, u);
  if (!f)
    f = (a = (i = (n = (r = ut()).handleMissingMessage) == null ? void 0 : n.call(r, { locale: u, id: e, defaultValue: l })) != null ? i : l) != null ? a : e;
  else if (typeof f != "string")
    return console.warn(
      `[svelte-i18n] Message with id "${e}" must be of type "string", found: "${typeof f}". Gettin its value through the "$format" method is deprecated; use the "json" method instead.`
    ), f;
  if (!s)
    return f;
  let p = f;
  try {
    p = Bo(f, u).format(s);
  } catch (v) {
    v instanceof Error && console.warn(
      `[svelte-i18n] Message "${e}" has syntax error:`,
      v.message
    );
  }
  return p;
}, Oo = (e, t) => Po(t).format(e), Mo = (e, t) => Ho(t).format(e), No = (e, t) => Ao(t).format(e), Lo = (e, t = Ke()) => oi(e, t);
ft([ht, Ht], () => Io);
ft([ht], () => Oo);
ft([ht], () => Mo);
ft([ht], () => No);
ft([ht, Ht], () => Lo);
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
  #t = X(wt({}));
  get shared() {
    return h(this.#t);
  }
  set shared(t) {
    w(this.#t, t, !0);
  }
  #r = X(wt({}));
  get props() {
    return h(this.#r);
  }
  set props(t) {
    w(this.#r, t, !0);
  }
  #e = X((t) => t);
  get i18n() {
    return h(this.#e);
  }
  set i18n(t) {
    w(this.#e, t, !0);
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
    ), be(() => {
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
    }), Object.keys(this.translatable_props).length > 0 && ht.subscribe(() => {
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
    be(() => {
      this.mounted || (this.old_value = this.props.value, this.mounted = !0), this.old_value != this.props.value && (this.old_value = this.props.value, this.dispatch("change"));
    });
  }
}
zt(["click"]);
function hr(e) {
  let t = ["", "k", "M", "G", "T", "P", "E", "Z"], r = 0;
  for (; e > 1e3 && r < t.length - 1; )
    e /= 1e3, r++;
  let n = t[r];
  return (Number.isInteger(e) ? e : e.toFixed(1)) + n;
}
function Tn(e) {
  return Object.prototype.toString.call(e) === "[object Date]";
}
function Br(e, t, r, n) {
  if (typeof r == "number" || Tn(r)) {
    const i = n - r, a = (r - t) / (e.dt || 1 / 60), o = e.opts.stiffness * i, s = e.opts.damping * a, u = (o - s) * e.inv_mass, l = (a + u) * e.dt;
    return Math.abs(l) < e.opts.precision && Math.abs(i) < e.opts.precision ? n : (e.settled = !1, Tn(r) ? new Date(r.getTime() + l) : r + l);
  } else {
    if (Array.isArray(r))
      return r.map(
        (i, a) => (
          // @ts-ignore
          Br(e, t[a], r[a], n[a])
        )
      );
    if (typeof r == "object") {
      const i = {};
      for (const a in r)
        i[a] = Br(e, t[a], r[a], n[a]);
      return i;
    } else
      throw new Error(`Cannot spring ${typeof r} values`);
  }
}
function Sn(e, t = {}) {
  const r = At(e), { stiffness: n = 0.15, damping: i = 0.8, precision: a = 0.01 } = t;
  let o, s, u, l = (
    /** @type {T} */
    e
  ), f = (
    /** @type {T | undefined} */
    e
  ), p = 1, v = 0, E = !1;
  function m(T, d = {}) {
    f = T;
    const c = u = {};
    return e == null || d.hard || y.stiffness >= 1 && y.damping >= 1 ? (E = !0, o = _e.now(), l = T, r.set(e = f), Promise.resolve()) : (d.soft && (v = 1 / ((d.soft === !0 ? 0.5 : +d.soft) * 60), p = 0), s || (o = _e.now(), E = !1, s = Na((_) => {
      if (E)
        return E = !1, s = null, !1;
      p = Math.min(p + v, 1);
      const g = Math.min(_ - o, 1e3 / 30), b = {
        inv_mass: p,
        opts: y,
        settled: !0,
        dt: g * 60 / 1e3
      }, H = Br(b, l, e, f);
      return o = _, l = /** @type {T} */
      e, r.set(e = /** @type {T} */
      H), b.settled && (s = null), !b.settled;
    })), new Promise((_) => {
      s.promise.then(() => {
        c === u && _();
      });
    }));
  }
  const y = {
    set: m,
    update: (T, d) => m(T(
      /** @type {T} */
      f,
      /** @type {T} */
      e
    ), d),
    subscribe: r.subscribe,
    stiffness: n,
    damping: i,
    precision: a
  };
  return y;
}
var Fo = /* @__PURE__ */ re('<div><svg viewBox="-1200 -1200 3000 3000" fill="none" xmlns="http://www.w3.org/2000/svg" class="svelte-m6d381"><g><path d="M255.926 0.754768L509.702 139.936V221.027L255.926 81.8465V0.754768Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M509.69 139.936L254.981 279.641V361.255L509.69 221.55V139.936Z" fill="#FF7C00" class="svelte-m6d381"></path><path d="M0.250138 139.937L254.981 279.641V361.255L0.250138 221.55V139.937Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M255.923 0.232622L0.236328 139.936V221.55L255.923 81.8469V0.232622Z" fill="#FF7C00" class="svelte-m6d381"></path></g><g><path d="M255.926 141.5L509.702 280.681V361.773L255.926 222.592V141.5Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M509.69 280.679L254.981 420.384V501.998L509.69 362.293V280.679Z" fill="#FF7C00" class="svelte-m6d381"></path><path d="M0.250138 280.681L254.981 420.386V502L0.250138 362.295V280.681Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M255.923 140.977L0.236328 280.68V362.294L255.923 222.591V140.977Z" fill="#FF7C00" class="svelte-m6d381"></path></g></svg></div>');
function Go(e, t) {
  qt(t, !0);
  const r = () => Qr(u, "$top", i), n = () => Qr(l, "$bottom", i), [i, a] = xa();
  var o = this && this.__awaiter || function(_, g, b, H) {
    function P(B) {
      return B instanceof b ? B : new b(function(M) {
        M(B);
      });
    }
    return new (b || (b = Promise))(function(B, M) {
      function D(ie) {
        try {
          ne(H.next(ie));
        } catch (U) {
          M(U);
        }
      }
      function J(ie) {
        try {
          ne(H.throw(ie));
        } catch (U) {
          M(U);
        }
      }
      function ne(ie) {
        ie.done ? B(ie.value) : P(ie.value).then(D, J);
      }
      ne((H = H.apply(_, g || [])).next());
    });
  };
  let s = S(t, "margin", 3, !0);
  const u = Sn([0, 0]), l = Sn([0, 0]);
  let f = X(!1);
  function p() {
    return o(this, void 0, void 0, function* () {
      yield Promise.all([u.set([125, 140]), l.set([-125, -140])]), yield Promise.all([u.set([-125, 140]), l.set([125, -140])]), yield Promise.all([u.set([-125, 0]), l.set([125, -0])]), yield Promise.all([u.set([125, 0]), l.set([-125, 0])]);
    });
  }
  function v() {
    return o(this, void 0, void 0, function* () {
      yield p(), h(f) || v();
    });
  }
  function E() {
    return o(this, void 0, void 0, function* () {
      yield Promise.all([u.set([125, 0]), l.set([-125, 0])]), v();
    });
  }
  be(() => (E(), () => {
    w(f, !0);
  }));
  var m = Fo();
  let y;
  var T = Y(m), d = Y(T), c = F(d);
  W(() => {
    y = Re(m, 1, "svelte-m6d381", null, y, { margin: s() }), ye(d, `transform: translate(${r()[0] ?? ""}px, ${r()[1] ?? ""}px);`), ye(c, `transform: translate(${n()[0] ?? ""}px, ${n()[1] ?? ""}px);`);
  }), L(e, m), Xt(), a();
}
var jo = function(e, t, r, n) {
  function i(a) {
    return a instanceof r ? a : new r(function(o) {
      o(a);
    });
  }
  return new (r || (r = Promise))(function(a, o) {
    function s(f) {
      try {
        l(n.next(f));
      } catch (p) {
        o(p);
      }
    }
    function u(f) {
      try {
        l(n.throw(f));
      } catch (p) {
        o(p);
      }
    }
    function l(f) {
      f.done ? a(f.value) : i(f.value).then(s, u);
    }
    l((n = n.apply(e, t || [])).next());
  });
};
let Lt = [], cr = !1;
const Vo = typeof window < "u", di = Vo ? window.requestAnimationFrame : (e) => {
};
function zo(e) {
  return jo(this, arguments, void 0, function* (t, r = !0) {
    if (!(window.__gradio_mode__ === "website" || window.__gradio_mode__ !== "app" && r !== !0)) {
      if (Lt.push(t), !cr) cr = !0;
      else return;
      yield va(), di(() => {
        let n = [0, 0];
        for (let i = 0; i < Lt.length; i++) {
          const o = Lt[i].getBoundingClientRect();
          (i === 0 || o.top + window.scrollY <= n[0]) && (n[0] = o.top + window.scrollY, n[1] = i);
        }
        window.scrollTo({ top: n[0] - 20, behavior: "smooth" }), cr = !1, Lt = [];
      });
    }
  });
}
var Xo = /* @__PURE__ */ re('<div class="validation-error svelte-124hqw6"> <button class="svelte-124hqw6"><!></button></div>'), qo = /* @__PURE__ */ re('<div class="eta-bar svelte-124hqw6"></div>'), Wo = /* @__PURE__ */ re("<!> ", 1), Zo = /* @__PURE__ */ re("<!> <!> <!> <!>", 1), Yo = /* @__PURE__ */ re('<div class="progress-level svelte-124hqw6"><div class="progress-level-inner svelte-124hqw6"><!></div> <div class="progress-bar-wrap svelte-124hqw6"><div class="progress-bar svelte-124hqw6"></div></div></div>'), Jo = /* @__PURE__ */ re('<p class="loading svelte-124hqw6"> </p> <!>', 1), Qo = /* @__PURE__ */ re("<!> <div><!> <!></div> <!> <!>", 1), Ko = /* @__PURE__ */ re('<div class="clear-status svelte-124hqw6"><!></div> <span class="error svelte-124hqw6"> </span> <!>', 1), $o = /* @__PURE__ */ re("<div> <!> </div>"), el = /* @__PURE__ */ re('<div data-testid="status-tracker"><!> <!></div> <!>', 1);
function tl(e, t) {
  qt(t, !0);
  let r = S(t, "eta", 3, null), n = S(t, "scroll_to_output", 3, !1), i = S(t, "timer", 3, !0), a = S(t, "show_progress", 3, "full"), o = S(t, "message", 3, null), s = S(t, "progress", 3, null), u = S(t, "variant", 3, "default"), l = S(t, "loading_text", 3, "Loading..."), f = S(t, "absolute", 3, !0), p = S(t, "translucent", 3, !1), v = S(t, "border", 3, !1), E = S(t, "validation_error", 7, null), m = S(t, "show_validation_error", 3, !0), y = S(t, "type", 3, null), T = S(t, "used_cache", 3, null), d = S(t, "cache_duration", 3, null), c = S(t, "avg_time", 3, null), _, g = !1, b = X(0), H = X(null), P = X(null), B = X(!1), M = X(null), D = X(!1), J = X(!1), ne = X(null), ie = X(null), U = X("from cache"), De = X(!1), xe = null, Ee = null;
  const Ve = ge(() => !(m() && E()) && (y() === "input" || !t.status || t.status === "complete" || a() === "hidden" || t.status == "streaming"));
  let we = X(0);
  const $e = ge(() => h(P) === null || h(P) <= 0 || !h(we) ? 0 : Math.min(h(we) / h(P), 1)), K = ge(() => h(we).toFixed(1));
  let Pe = ge(() => s() == null), de = ge(() => r() !== null && r() !== void 0 ? r() : h(H));
  function Be() {
    di(() => {
      w(we, (performance.now() - h(b)) / 1e3), g && Be();
    });
  }
  let $ = ge(() => {
    let G = null;
    s() != null ? G = s().map((V) => {
      if (V.index != null && V.length != null)
        return V.index / V.length;
      if (V.progress != null)
        return V.progress;
    }) : G = null;
    let Z, Q = "";
    return G ? (Z = G[G.length - 1], Z === 0 ? Q = "0" : Q = "150ms") : Z = void 0, {
      progress_level: G,
      last_progress_level: Z,
      progress_bar_transition: Q
    };
  });
  function ee() {
    g || (w(H, w(M, null), !0), w(b, performance.now(), !0), g = !0, Be());
  }
  function ke() {
    w(H, w(M, null), !0), g && (g = !1);
  }
  be(() => {
    t.status === "pending" ? ee() : te(() => {
      ke();
    });
  }), be(() => {
    _ && n() && (t.status === "pending" || t.status === "complete") && zo(_, t.autoscroll);
  }), be(() => {
    h(de) != null && h(H) !== h(de) && (w(P, (performance.now() - h(b)) / 1e3 + h(de)), w(M, h(P).toFixed(1), !0), w(H, h(de), !0));
  });
  function Ie() {
    w(B, !1);
  }
  be(() => {
    te(() => {
      Ie();
    }), t.status === "error" && o() && w(B, !0);
  }), be(() => {
    t.status === "complete" && y() === "output" && T() && d() != null && (w(ne, d().toFixed(1), !0), w(U, T() === "full" ? "from cache" : "used cache", !0), w(De, c() != null && c() > d() && c() > 0, !0), w(ie, h(De) ? c().toFixed(1) : null, !0), w(D, !0), w(J, !1), xe && clearTimeout(xe), Ee && clearTimeout(Ee), xe = setTimeout(
      () => {
        w(J, !0), Ee = setTimeout(
          () => {
            w(D, !1), w(J, !1);
          },
          500
        );
      },
      1750
    ));
  });
  var ae = el(), x = fe(ae);
  let I, A;
  var O = Y(x);
  {
    var se = (G) => {
      var Z = Xo(), Q = Y(Z), V = F(Q), ce = Y(V);
      {
        let me = ge(() => t.i18n ? t.i18n("common.clear") : "Clear");
        ln(ce, {
          get Icon() {
            return un;
          },
          get label() {
            return h(me);
          },
          disabled: !1,
          size: "x-small",
          background: "var(--background-fill-primary)",
          color: "var(--error-background-text)",
          border: "var(--border-color-primary)",
          onclick: () => E(null)
        });
      }
      W(() => ue(Q, `${E() ?? ""} `)), L(G, Z);
    };
    z(O, (G) => {
      E() && m() && G(se);
    });
  }
  var Oe = F(O, 2);
  {
    var Ue = (G) => {
      var Z = Qo(), Q = fe(Z);
      {
        var V = (k) => {
          var q = qo();
          let Se;
          W(() => Se = ye(q, "", Se, {
            transform: `translateX(${(h($e) || 0) * 100 - 100}%)`
          })), L(k, q);
        };
        z(Q, (k) => {
          u() === "default" && h(Pe) && a() === "full" && k(V);
        });
      }
      var ce = F(Q, 2);
      let me;
      var Te = Y(ce);
      {
        var Xe = (k) => {
          var q = it(), Se = fe(q);
          en(Se, 17, s, Kr, (pt, Ae) => {
            var Bt = it(), Kt = fe(Bt);
            {
              var It = (qe) => {
                var vt = Wo(), Ot = fe(vt);
                {
                  var $t = (Me) => {
                    var We = He();
                    W((mt, gt) => ue(We, `${mt ?? ""}/${gt ?? ""}`), [
                      () => hr(h(Ae).index || 0),
                      () => hr(h(Ae).length)
                    ]), L(Me, We);
                  }, et = (Me) => {
                    var We = He();
                    W((mt) => ue(We, mt), [() => hr(h(Ae).index || 0)]), L(Me, We);
                  };
                  z(Ot, (Me) => {
                    h(Ae).length != null ? Me($t) : Me(et, -1);
                  });
                }
                var tt = F(Ot);
                W(() => ue(tt, ` ${h(Ae).unit ?? ""} |  `)), L(qe, vt);
              };
              z(Kt, (qe) => {
                h(Ae).index != null && qe(It);
              });
            }
            L(pt, Bt);
          }), L(k, q);
        }, dt = (k) => {
          var q = He();
          W(() => ue(q, `queue: ${t.queue_position + 1}/${t.queue_size ?? ""} |`)), L(k, q);
        }, Pt = (k) => {
          var q = He("processing |");
          L(k, q);
        };
        z(Te, (k) => {
          s() ? k(Xe) : t.queue_position !== null && t.queue_size !== void 0 && t.queue_position >= 0 ? k(dt, 1) : t.queue_position === 0 && k(Pt, 2);
        });
      }
      var pi = F(Te, 2);
      {
        var vi = (k) => {
          var q = He();
          W(() => ue(q, `${h(K) ?? ""}${r() ? `/${h(M)}` : ""}s`)), L(k, q);
        };
        z(pi, (k) => {
          i() && k(vi);
        });
      }
      var Gr = F(ce, 2);
      {
        var mi = (k) => {
          var q = Yo(), Se = Y(q), pt = Y(Se);
          {
            var Ae = (qe) => {
              var vt = it(), Ot = fe(vt);
              en(Ot, 17, s, Kr, ($t, et, tt) => {
                var Me = it(), We = fe(Me);
                {
                  var mt = (gt) => {
                    var jr = Zo(), Vr = fe(jr);
                    {
                      var yi = (le) => {
                        var Ne = He(" /");
                        L(le, Ne);
                      };
                      z(Vr, (le) => {
                        tt !== 0 && le(yi);
                      });
                    }
                    var zr = F(Vr, 2);
                    {
                      var xi = (le) => {
                        var Ne = He();
                        W(() => ue(Ne, h(et).desc)), L(le, Ne);
                      };
                      z(zr, (le) => {
                        h(et).desc != null && le(xi);
                      });
                    }
                    var Xr = F(zr, 2);
                    {
                      var Ei = (le) => {
                        var Ne = He("-");
                        L(le, Ne);
                      };
                      z(Xr, (le) => {
                        h(et).desc != null && h($).progress_level && h($).progress_level[tt] != null && le(Ei);
                      });
                    }
                    var wi = F(Xr, 2);
                    {
                      var Ti = (le) => {
                        var Ne = He();
                        W((Si) => ue(Ne, `${Si ?? ""}%`), [
                          () => (100 * (h($).progress_level[tt] || 0)).toFixed(1)
                        ]), L(le, Ne);
                      };
                      z(wi, (le) => {
                        h($).progress_level != null && le(Ti);
                      });
                    }
                    L(gt, jr);
                  };
                  z(We, (gt) => {
                    (h(et).desc != null || h($).progress_level && h($).progress_level[tt] != null) && gt(mt);
                  });
                }
                L($t, Me);
              }), L(qe, vt);
            };
            z(pt, (qe) => {
              s() != null && qe(Ae);
            });
          }
          var Bt = F(Se, 2), Kt = Y(Bt);
          let It;
          W(() => It = ye(Kt, "", It, {
            width: `${h($).last_progress_level * 100}%`,
            transition: h($).progress_bar_transition
          })), L(k, q);
        }, gi = (k) => {
          {
            let q = ge(() => u() === "default");
            Go(k, {
              get margin() {
                return h(q);
              }
            });
          }
        };
        z(Gr, (k) => {
          h($).last_progress_level != null ? k(mi) : a() === "full" && k(gi, 1);
        });
      }
      var bi = F(Gr, 2);
      {
        var _i = (k) => {
          var q = Jo(), Se = fe(q), pt = Y(Se), Ae = F(Se, 2);
          br(Ae, t, "additional-loading-text", {}), W(() => ue(pt, l())), L(k, q);
        };
        z(bi, (k) => {
          i() || k(_i);
        });
      }
      W(() => me = Re(ce, 1, "progress-text svelte-124hqw6", null, me, {
        "meta-text-center": u() === "center",
        "meta-text": u() === "default"
      })), L(G, Z);
    }, ve = (G) => {
      var Z = Ko(), Q = fe(Z), V = Y(Q);
      {
        let Xe = ge(() => t.i18n("common.clear"));
        ln(V, {
          get Icon() {
            return un;
          },
          get label() {
            return h(Xe);
          },
          disabled: !1,
          $$events: {
            click: () => {
              t.on_clear_status?.();
            }
          }
        });
      }
      var ce = F(Q, 2), me = Y(ce), Te = F(ce, 2);
      br(Te, t, "error", {}), W((Xe) => ue(me, Xe), [() => t.i18n("common.error")]), L(G, Z);
    };
    z(Oe, (G) => {
      t.status === "pending" ? G(Ue) : t.status === "error" && G(ve, 1);
    });
  }
  kr(x, (G) => _ = G, () => _);
  var oe = F(x, 2);
  {
    var ze = (G) => {
      var Z = $o();
      let Q, V;
      var ce = Y(Z), me = F(ce);
      {
        var Te = (dt) => {
          var Pt = He();
          W(() => ue(Pt, `~${h(ie) ?? ""}s
			→ `)), L(dt, Pt);
        };
        z(me, (dt) => {
          h(De) && dt(Te);
        });
      }
      var Xe = F(me);
      W(() => {
        Q = Re(Z, 1, "cache-indicator svelte-124hqw6", null, Q, { "fade-out": h(J) }), V = ye(Z, "", V, { position: f() ? "absolute" : "static" }), ue(ce, `⚡ ${h(U) ?? ""}: `), ue(Xe, `${h(ne) ?? ""}s`);
      }), L(G, Z);
    };
    z(oe, (G) => {
      h(D) && G(ze);
    });
  }
  W(() => {
    I = Re(x, 1, `wrap ${u() ?? ""} ${a() ?? ""}`, "svelte-124hqw6", I, {
      "no-click": E() && m(),
      hide: h(Ve),
      translucent: u() === "center" && (t.status === "pending" || t.status === "error") || p() || a() === "minimal" || E(),
      generating: t.status === "generating" && a() === "full",
      border: v()
    }), A = ye(x, "", A, {
      position: f() ? "absolute" : "static",
      padding: f() ? "0" : "var(--size-8) 0"
    });
  }), L(e, ae), Xt();
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
zt(["touchstart", "touchmove", "touchend", "click", "keydown"]);
var sl = /* @__PURE__ */ new Set(["$$slots", "$$events", "$$legacy"]), ol = /* @__PURE__ */ re('<!> <div class="region-annotator svelte-r41nsf"><div class="toolbar svelte-r41nsf" role="group" aria-label="Region annotation tool"><button type="button">浏览</button> <button type="button">套索选择</button> <button type="button" class="clear svelte-r41nsf">清除 Draft</button></div> <div class="legend svelte-r41nsf"><span class="svelte-r41nsf"><i class="draft svelte-r41nsf"></i>黄色：Draft</span> <span class="svelte-r41nsf"><i class="saved svelte-r41nsf"></i>绿色：Saved Region</span></div> <div class="canvas-wrap svelte-r41nsf"><canvas class="svelte-r41nsf"></canvas></div> <div class="status svelte-r41nsf"> </div></div>', 1);
function ul(e, t) {
  qt(t, !0);
  const r = /* @__PURE__ */ Ya(t, sl), n = new Uo(r), i = 2048, a = 4096, o = 3;
  let s, u = null, l = null, f = null, p = null, v = null, E = X(wt({})), m = X("browse"), y = X("请先生成版图 binary mask"), T = X(!1), d = X(!1), c = X(wt([])), _ = null, g = null, b = "";
  function H(x) {
    return JSON.parse(JSON.stringify(x || {}));
  }
  function P(x) {
    return typeof x == "number" ? `${x}px` : x || "520px";
  }
  function B(x, I, A) {
    return Math.max(I, Math.min(A, x));
  }
  function M() {
    return h(E).server_view || {};
  }
  function D() {
    return h(E).client_intent || {};
  }
  function J() {
    return Math.max(1, Number(M().natural_width || u?.naturalWidth || 1));
  }
  function ne() {
    return Math.max(1, Number(M().natural_height || u?.naturalHeight || 1));
  }
  function ie() {
    return Math.min(1, i / Math.max(J(), ne()));
  }
  function U(x, I) {
    if (!x) {
      I(null);
      return;
    }
    const A = new Image();
    A.onload = () => I(A), A.onerror = () => I(null), A.src = x;
  }
  function De() {
    if (!l) {
      f = null;
      return;
    }
    const x = J(), I = ne(), A = document.createElement("canvas");
    A.width = x, A.height = I;
    const O = A.getContext("2d", { willReadFrequently: !0 });
    if (!O) return;
    O.imageSmoothingEnabled = !1, O.drawImage(l, 0, 0, x, I);
    const se = O.getImageData(0, 0, x, I), Oe = document.createElement("canvas");
    Oe.width = x, Oe.height = I;
    const Ue = Oe.getContext("2d");
    if (!Ue) return;
    const ve = Ue.createImageData(x, I);
    for (let oe = 0; oe < se.data.length; oe += 4) {
      const ze = Math.max(se.data[oe], se.data[oe + 1], se.data[oe + 2]);
      se.data[oe + 3] > 0 && ze >= 128 && (ve.data[oe] = 45, ve.data[oe + 1] = 160, ve.data[oe + 2] = 255, ve.data[oe + 3] = 80);
    }
    Ue.putImageData(ve, 0, 0), f = Oe;
  }
  function xe(x) {
    w(E, H(x), !0);
    const I = D();
    w(m, I.tool_mode === "lasso" ? "lasso" : "browse", !0), w(
      c,
      Array.isArray(I.lasso_polygon) ? I.lasso_polygon.filter((A) => Array.isArray(A) && A.length === 2).map((A) => ({ x: Number(A[0]), y: Number(A[1]) })) : [],
      !0
    ), w(y, M().status || "Region 标注器已加载", !0), w(T, !1), w(d, !1), _ = null, g = null, U(M().source_image, (A) => {
      u = A, ae();
    }), U(M().source_mask_image, (A) => {
      l = A, De(), ae();
    }), U(M().saved_region_overlay_image, (A) => {
      p = A, ae();
    }), U(M().draft_region_overlay_image, (A) => {
      v = A, ae();
    });
  }
  be(() => {
    const x = JSON.stringify(n.props.value || null);
    x !== b && (b = x, xe(n.props.value));
  });
  function Ee(x = !1) {
    w(
      E,
      Object.assign(Object.assign({}, h(E)), {
        client_intent: Object.assign(Object.assign({}, D()), {
          tool_mode: h(m),
          lasso_polygon: h(c).map((I) => [I.x, I.y])
        })
      }),
      !0
    ), n.props.value = h(E), b = JSON.stringify(h(E)), x && n.dispatch("input");
  }
  function Ve(x) {
    h(T) && ee("绘制已取消"), w(m, x, !0), w(y, x === "lasso" ? "按住鼠标左键拖动套索" : "浏览模式：Canvas 只读", !0), Ee(!1), ae();
  }
  function we(x) {
    const I = s.getBoundingClientRect();
    return {
      x: B((x.clientX - I.left) / Math.max(1, I.width) * J(), 0, J() - 1),
      y: B((x.clientY - I.top) / Math.max(1, I.height) * ne(), 0, ne() - 1)
    };
  }
  function $e() {
    w(
      E,
      Object.assign(Object.assign({}, h(E)), {
        server_view: Object.assign(Object.assign({}, M()), { draft_region_overlay_image: "" })
      }),
      !0
    ), v = null;
  }
  function K(x) {
    x.button !== 0 || h(m) !== "lasso" || M().enabled === !1 || (x.preventDefault(), $e(), w(c, [we(x)], !0), w(T, !0), w(d, !1), _ = x.pointerId, g = { x: x.clientX, y: x.clientY }, s.setPointerCapture(x.pointerId), w(y, "正在绘制 Draft"), Ee(!1), ae());
  }
  function Pe(x) {
    if (!h(T) || x.pointerId !== _ || !g || (x.preventDefault(), Math.hypot(x.clientX - g.x, x.clientY - g.y) < o)) return;
    const A = we(x);
    h(c).length < a ? w(c, [...h(c), A], !0) : (w(c, [...h(c).slice(0, -1), A], !0), w(d, !0)), g = { x: x.clientX, y: x.clientY }, w(
      y,
      h(d) ? `已达到 ${a} 点上限` : `Draft 点数：${h(c).length}`,
      !0
    ), ae();
  }
  function de(x) {
    const I = we(x), A = h(c)[h(c).length - 1];
    A && A.x === I.x && A.y === I.y || (h(c).length < a ? w(c, [...h(c), I], !0) : w(c, [...h(c).slice(0, -1), I], !0));
  }
  function Be(x) {
    _ !== null && s.hasPointerCapture(_) && s.releasePointerCapture(_), _ = null, g = null, w(T, !1), x.preventDefault();
  }
  function $(x) {
    if (!h(T) || x.pointerId !== _) return;
    de(x), Be(x);
    const I = new Set(h(c).map((A) => `${A.x.toFixed(4)},${A.y.toFixed(4)}`));
    if (h(c).length < 3 || I.size < 3) {
      ee("套索点不足，Draft 已清除");
      return;
    }
    w(y, "正在生成权威 Draft 交集预览…"), Ee(!0), ae();
  }
  function ee(x = "Draft 已清除") {
    _ !== null && s?.hasPointerCapture(_) && s.releasePointerCapture(_), _ = null, g = null, w(T, !1), w(d, !1), w(c, [], !0), $e(), w(y, x, !0), Ee(!1), ae();
  }
  function ke(x) {
    x.pointerId === _ && ee("指针操作已取消，Draft 已清除");
  }
  function Ie() {
    h(T) && ee("窗口失焦，Draft 已清除");
  }
  function ae() {
    if (!s) return;
    const x = J(), I = ne(), A = ie();
    s.width = Math.max(1, Math.round(x * A)), s.height = Math.max(1, Math.round(I * A));
    const O = s.getContext("2d");
    if (O && (O.setTransform(A, 0, 0, A, 0, 0), O.clearRect(0, 0, x, I), u ? (O.imageSmoothingEnabled = !0, O.drawImage(u, 0, 0, x, I)) : (O.fillStyle = "#f8fafc", O.fillRect(0, 0, x, I), O.fillStyle = "#64748b", O.font = "18px sans-serif", O.fillText("请先生成版图 binary mask", 24, 42)), O.imageSmoothingEnabled = !1, f && O.drawImage(f, 0, 0, x, I), p && O.drawImage(p, 0, 0, x, I), v && O.drawImage(v, 0, 0, x, I), h(c).length > 0)) {
      O.save(), O.beginPath(), O.moveTo(h(c)[0].x, h(c)[0].y);
      for (let se = 1; se < h(c).length; se += 1) O.lineTo(h(c)[se].x, h(c)[se].y);
      !h(T) && h(c).length >= 3 && O.closePath(), O.setLineDash([8, 5]), O.lineWidth = Math.max(2, x / 700), O.strokeStyle = "rgba(255, 220, 0, 0.98)", O.stroke(), O.restore();
    }
  }
  Le("blur", ma, Ie);
  {
    let x = ge(() => h(T) ? "focus" : "base");
    es(e, {
      get visible() {
        return n.shared.visible;
      },
      variant: "solid",
      get border_mode() {
        return h(x);
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
      children: (I, A) => {
        var O = ol(), se = fe(O);
        tl(se, Qa(
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
        var Oe = F(se, 2), Ue = Y(Oe), ve = Y(Ue);
        let oe;
        var ze = F(ve, 2);
        let G;
        var Z = F(ze, 2), Q = F(Ue, 4), V = Y(Q);
        kr(V, (Te) => s = Te, () => s);
        var ce = F(Q, 2), me = Y(ce);
        W(
          (Te) => {
            ye(Oe, Te), oe = Re(ve, 1, "svelte-r41nsf", null, oe, { active: h(m) === "browse" }), G = Re(ze, 1, "svelte-r41nsf", null, G, { active: h(m) === "lasso" }), ye(V, `cursor:${h(m) === "lasso" ? "crosshair" : "default"}`), ue(me, `${h(y) ?? ""} · 点数 ${h(c).length ?? ""}/4096`);
          },
          [() => `min-height:${P(n.props.height)}`]
        ), Le("click", ve, () => Ve("browse")), Le("click", ze, () => Ve("lasso")), Le("click", Z, () => ee()), Le("pointerdown", V, K), Le("pointermove", V, Pe), Le("pointerup", V, $), Le("pointercancel", V, ke), L(I, O);
      },
      $$slots: { default: !0 }
    });
  }
  Xt();
}
export {
  ul as default
};

import { i as Cn, g as Ri, o as Ja, n as xt, u as xe, s as Qa, r as gn, m as Mt, a as x, b as o, t as Rn, d as Ka, q as ki, c as Di, e as Et, f as Dr, h as Lr, j as $a, T as es, k as ts, l as Br, p as Pt, v as kn, w as Ot, x as Gi, y as Fi, z as Ui, A as ur, E as Gr, B as jt, C as ji, D as We, F as ai, G as rs, H as Vi, I as Dn, J as ns, K as si, L as is, M as as, N as lt, O as zi, P as nn, Q as ss, R as os, S as ls, U as Xi, V as us, W as fs, X as Wi, Y as Gn, Z as oi, _ as li, $ as cs, a0 as hs, a1 as ds, a2 as ps, a3 as vs, a4 as ms, a5 as gs, a6 as bs, a7 as Fn, a8 as _s, a9 as qi, aa as Fr, ab as ys, ac as xs, ad as Es, ae as ws, af as Ts, ag as Ss, ah as Un, ai as As, aj as Hs, ak as ze, al as bn, am as _n, an as Ms, ao as Ps, ap as or, aq as Os, ar as Ns, as as Is, at as Ls, au as Bs, av as Zi, aw as tr, ax as Y, ay as Cs, az as ui, aA as Rs, aB as Ne, aC as Ur, aD as jr, aE as z, aF as Oe, aG as oe, aH as ks, aI as te, aJ as ge, aK as Xe, aL as Ds } from "./render-CPsC5Y6J.js";
function Yi(e) {
  throw new Error("https://svelte.dev/e/lifecycle_outside_component");
}
const Gs = [];
function Fs(e, t = !1, r = !1) {
  return Or(e, /* @__PURE__ */ new Map(), "", Gs, null, r);
}
function Or(e, t, r, n, i = null, s = !1) {
  if (typeof e == "object" && e !== null) {
    var u = t.get(e);
    if (u !== void 0) return u;
    if (e instanceof Map) return (
      /** @type {Snapshot<T>} */
      new Map(e)
    );
    if (e instanceof Set) return (
      /** @type {Snapshot<T>} */
      new Set(e)
    );
    if (Cn(e)) {
      var f = (
        /** @type {Snapshot<any>} */
        Array(e.length)
      );
      t.set(e, f), i !== null && t.set(i, f);
      for (var h = 0; h < e.length; h += 1) {
        var c = e[h];
        h in e && (f[h] = Or(c, t, r, n, null, s));
      }
      return f;
    }
    if (Ri(e) === Ja) {
      f = {}, t.set(e, f), i !== null && t.set(i, f);
      for (var d of Object.keys(e))
        f[d] = Or(
          // @ts-expect-error
          e[d],
          t,
          r,
          n,
          null,
          s
        );
      return f;
    }
    if (e instanceof Date)
      return (
        /** @type {Snapshot<T>} */
        structuredClone(e)
      );
    if (typeof /** @type {T & { toJSON?: any } } */
    e.toJSON == "function" && !s)
      return Or(
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
function jn(e, t, r) {
  if (e == null)
    return t(void 0), r && r(void 0), xt;
  const n = xe(
    () => e.subscribe(
      t,
      // @ts-expect-error
      r
    )
  );
  return n.unsubscribe ? () => n.unsubscribe() : n;
}
const Gt = [];
function Us(e, t) {
  return {
    subscribe: fr(e, t).subscribe
  };
}
function fr(e, t = xt) {
  let r = null;
  const n = /* @__PURE__ */ new Set();
  function i(f) {
    if (Qa(e, f) && (e = f, r)) {
      const h = !Gt.length;
      for (const c of n)
        c[1](), Gt.push(c, e);
      if (h) {
        for (let c = 0; c < Gt.length; c += 2)
          Gt[c][0](Gt[c + 1]);
        Gt.length = 0;
      }
    }
  }
  function s(f) {
    i(f(
      /** @type {T} */
      e
    ));
  }
  function u(f, h = xt) {
    const c = [f, h];
    return n.add(c), n.size === 1 && (r = t(i, s) || xt), f(
      /** @type {T} */
      e
    ), () => {
      n.delete(c), n.size === 0 && r && (r(), r = null);
    };
  }
  return { set: i, update: s, subscribe: u };
}
function qt(e, t, r) {
  const n = !Array.isArray(e), i = n ? [e] : e;
  if (!i.every(Boolean))
    throw new Error("derived() expects stores as input, got a falsy value");
  const s = t.length < 2;
  return Us(r, (u, f) => {
    let h = !1;
    const c = [];
    let d = 0, _ = xt;
    const b = () => {
      if (d)
        return;
      _();
      const O = t(n ? c[0] : c, u, f);
      s ? u(O) : _ = typeof O == "function" ? O : xt;
    }, H = i.map(
      (O, w) => jn(
        O,
        (m) => {
          c[w] = m, d &= ~(1 << w), h && b();
        },
        () => {
          d |= 1 << w;
        }
      )
    );
    return h = !0, b(), function() {
      gn(H), _(), h = !1;
    };
  });
}
function js(e) {
  let t;
  return jn(e, (r) => t = r)(), t;
}
let Hr = !1, yn = /* @__PURE__ */ Symbol("unmounted");
function fi(e, t, r) {
  const n = r[t] ??= {
    store: null,
    source: Mt(void 0),
    unsubscribe: xt
  };
  if (n.store !== e && !(yn in r))
    if (n.unsubscribe(), n.store = e ?? null, e == null)
      n.source.v = void 0, n.unsubscribe = xt;
    else {
      var i = !0;
      n.unsubscribe = jn(e, (s) => {
        i ? n.source.v = s : x(n.source, s);
      }), i = !1;
    }
  return e && yn in r ? js(e) : o(n.source);
}
function Vs() {
  const e = {};
  function t() {
    Rn(() => {
      for (var r in e)
        e[r].unsubscribe();
      Ka(e, yn, {
        enumerable: !1,
        value: !0
      });
    });
  }
  return [e, t];
}
function zs(e) {
  var t = Hr;
  try {
    return Hr = !1, [e(), Hr];
  } finally {
    Hr = t;
  }
}
function Xs(e, t) {
  if (t) {
    const r = document.body;
    e.autofocus = !0, ki(() => {
      document.activeElement === r && e.focus();
    });
  }
}
const Ws = (
  // We gotta write it like this because after downleveling the pure comment may end up in the wrong location
  globalThis?.window?.trustedTypes && /* @__PURE__ */ globalThis.window.trustedTypes.createPolicy("svelte-trusted-html", {
    /** @param {string} html */
    createHTML: (e) => e
  })
);
function qs(e) {
  return (
    /** @type {string} */
    Ws?.createHTML(e) ?? e
  );
}
function Ji(e) {
  var t = Di("template");
  return t.innerHTML = qs(e.replaceAll("<!>", "<!---->")), t.content;
}
function Vt(e, t) {
  var r = (
    /** @type {Effect} */
    Dr
  );
  r.nodes === null && (r.nodes = { start: e, end: t, a: null, t: null });
}
// @__NO_SIDE_EFFECTS__
function Ee(e, t) {
  var r = (t & es) !== 0, n = (t & ts) !== 0, i, s = !e.startsWith("<!>");
  return () => {
    i === void 0 && (i = Ji(s ? e : "<!>" + e), r || (i = /** @type {TemplateNode} */
    Lr(i)));
    var u = (
      /** @type {TemplateNode} */
      n || $a ? document.importNode(i, !0) : i.cloneNode(!0)
    );
    if (r) {
      var f = (
        /** @type {TemplateNode} */
        Lr(u)
      ), h = (
        /** @type {TemplateNode} */
        u.lastChild
      );
      Vt(f, h);
    } else
      Vt(u, u);
    return u;
  };
}
// @__NO_SIDE_EFFECTS__
function Zs(e, t, r = "svg") {
  var n = !e.startsWith("<!>"), i = `<${r}>${n ? e : "<!>" + e}</${r}>`, s;
  return () => {
    if (!s) {
      var u = (
        /** @type {DocumentFragment} */
        Ji(i)
      ), f = (
        /** @type {Element} */
        Lr(u)
      );
      s = /** @type {Element} */
      Lr(f);
    }
    var h = (
      /** @type {TemplateNode} */
      s.cloneNode(!0)
    );
    return Vt(h, h), h;
  };
}
// @__NO_SIDE_EFFECTS__
function Qi(e, t) {
  return /* @__PURE__ */ Zs(e, t, "svg");
}
function ot(e = "") {
  {
    var t = Et(e + "");
    return Vt(t, t), t;
  }
}
function Ut() {
  var e = document.createDocumentFragment(), t = document.createComment(""), r = Et();
  return e.append(t, r), Vt(t, r), e;
}
function j(e, t) {
  e !== null && e.before(
    /** @type {Node} */
    t
  );
}
class Vr {
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
        Br(n), this.#n.delete(r);
      else {
        var i = this.#e.get(r);
        i && (Br(i.effect), this.#r.set(r, i.effect), this.#e.delete(r), i.fragment.lastChild.remove(), this.anchor.before(i.fragment), n = i.effect);
      }
      for (const [s, u] of this.#t) {
        if (this.#t.delete(s), s === t)
          break;
        const f = this.#e.get(u);
        f && (Pt(f.effect), this.#e.delete(u));
      }
      for (const [s, u] of this.#r) {
        if (s === r || this.#n.has(s)) continue;
        const f = () => {
          if (Array.from(this.#t.values()).includes(s)) {
            var c = document.createDocumentFragment();
            Fi(u, c), c.append(Et()), this.#e.set(s, { effect: u, fragment: c });
          } else
            Pt(u);
          this.#n.delete(s), this.#r.delete(s);
        };
        this.#i || !n ? (this.#n.add(s), kn(u, f, !1)) : f();
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
      r.includes(n) || (Pt(i.effect), this.#e.delete(n));
  };
  /**
   *
   * @param {any} key
   * @param {null | ((target: TemplateNode) => void)} fn
   */
  ensure(t, r) {
    var n = (
      /** @type {Batch} */
      Gi
    ), i = Ui();
    if (r && !this.#r.has(t) && !this.#e.has(t))
      if (i) {
        var s = document.createDocumentFragment(), u = Et();
        s.append(u), this.#e.set(t, {
          effect: Ot(() => r(u)),
          fragment: s
        });
      } else
        this.#r.set(
          t,
          Ot(() => r(this.anchor))
        );
    if (this.#t.set(n, t), i) {
      for (const [f, h] of this.#r)
        f === t ? n.unskip_effect(h) : n.skip_effect(h);
      for (const [f, h] of this.#e)
        f === t ? n.unskip_effect(h.effect) : n.skip_effect(h.effect);
      n.oncommit(this.#a), n.ondiscard(this.#s);
    } else
      this.#a(n);
  }
}
function Ys(e, t, ...r) {
  var n = new Vr(e);
  ur(() => {
    const i = t() ?? null;
    n.ensure(i, i && ((s) => i(s, ...r)));
  }, Gr);
}
function Js(e) {
  jt === null && Yi(), ji && jt.l !== null ? Ks(jt).m.push(e) : We(() => {
    const t = xe(e);
    if (typeof t == "function") return (
      /** @type {() => void} */
      t
    );
  });
}
function Qs(e) {
  jt === null && Yi(), Js(() => () => xe(e));
}
function Ks(e) {
  var t = (
    /** @type {ComponentContextLegacy} */
    e.l
  );
  return t.u ??= { a: [], b: [], m: [] };
}
function se(e, t, r = !1) {
  var n = new Vr(e), i = r ? Gr : 0;
  function s(u, f) {
    n.ensure(u, f);
  }
  ur(() => {
    var u = !1;
    t((f, h = 0) => {
      u = !0, s(h, f);
    }), u || s(-1, null);
  }, i);
}
function xn(e, t) {
  return t;
}
function $s(e, t, r) {
  for (var n = [], i = t.length, s, u = t.length, f = 0; f < i; f++) {
    let _ = t[f];
    kn(
      _,
      () => {
        if (s) {
          if (s.pending.delete(_), s.done.add(_), s.pending.size === 0) {
            var b = (
              /** @type {Set<EachOutroGroup>} */
              e.outrogroups
            );
            En(e, Dn(s.done)), b.delete(s), b.size === 0 && (e.outrogroups = null);
          }
        } else
          u -= 1;
      },
      !1
    );
  }
  if (u === 0) {
    var h = n.length === 0 && r !== null;
    if (h) {
      var c = (
        /** @type {Element} */
        r
      ), d = (
        /** @type {Element} */
        c.parentNode
      );
      os(d), d.append(c), e.items.clear();
    }
    En(e, t, !h);
  } else
    s = {
      pending: new Set(t),
      done: /* @__PURE__ */ new Set()
    }, (e.outrogroups ??= /* @__PURE__ */ new Set()).add(s);
}
function En(e, t, r = !0) {
  var n;
  if (e.pending.size > 0) {
    n = /* @__PURE__ */ new Set();
    for (const u of e.pending.values())
      for (const f of u)
        n.add(
          /** @type {EachItem} */
          e.items.get(f).e
        );
  }
  for (var i = 0; i < t.length; i++) {
    var s = t[i];
    if (n?.has(s)) {
      s.f |= lt;
      const u = document.createDocumentFragment();
      Fi(s, u);
    } else
      Pt(t[i], r);
  }
}
var ci;
function wn(e, t, r, n, i, s = null) {
  var u = e, f = /* @__PURE__ */ new Map(), h = (t & Xi) !== 0;
  if (h) {
    var c = (
      /** @type {Element} */
      e
    );
    u = c.appendChild(Et());
  }
  var d = null, _ = Vi(() => {
    var T = r();
    return (
      /** @type {V[]} */
      Cn(T) ? T : T == null ? [] : Dn(T)
    );
  }), b, H = /* @__PURE__ */ new Map(), O = !0;
  function w(T) {
    (E.effect.f & zi) === 0 && (E.pending.delete(T), E.fallback = d, eo(E, b, u, t, n), d !== null && (b.length === 0 ? (d.f & lt) === 0 ? Br(d) : (d.f ^= lt, ir(d, null, u)) : kn(d, () => {
      d = null;
    })));
  }
  function m(T) {
    E.pending.delete(T);
  }
  var v = ur(() => {
    b = /** @type {V[]} */
    o(_);
    for (var T = b.length, y = /* @__PURE__ */ new Set(), S = (
      /** @type {Batch} */
      Gi
    ), B = Ui(), N = 0; N < T; N += 1) {
      var k = b[N], U = n(k, N), R = O ? null : f.get(U);
      R ? (R.v && ai(R.v, k), R.i && ai(R.i, N), B && S.unskip_effect(R.e)) : (R = to(
        f,
        O ? u : ci ??= Et(),
        k,
        U,
        N,
        i,
        t,
        r
      ), O || (R.e.f |= lt), f.set(U, R)), y.add(U);
    }
    if (T === 0 && s && !d && (O ? d = Ot(() => s(u)) : (d = Ot(() => s(ci ??= Et())), d.f |= lt)), T > y.size && rs(), !O)
      if (H.set(S, y), B) {
        for (const [$, D] of f)
          y.has($) || S.skip_effect(D.e);
        S.oncommit(w), S.ondiscard(m);
      } else
        w(S);
    o(_);
  }), E = { effect: v, items: f, pending: H, outrogroups: null, fallback: d };
  O = !1;
}
function rr(e) {
  for (; e !== null && (e.f & ss) === 0; )
    e = e.next;
  return e;
}
function eo(e, t, r, n, i) {
  var s = (n & us) !== 0, u = t.length, f = e.items, h = rr(e.effect.first), c, d = null, _, b = [], H = [], O, w, m, v;
  if (s)
    for (v = 0; v < u; v += 1)
      O = t[v], w = i(O, v), m = /** @type {EachItem} */
      f.get(w).e, (m.f & lt) === 0 && (m.nodes?.a?.measure(), (_ ??= /* @__PURE__ */ new Set()).add(m));
  for (v = 0; v < u; v += 1) {
    if (O = t[v], w = i(O, v), m = /** @type {EachItem} */
    f.get(w).e, e.outrogroups !== null)
      for (const R of e.outrogroups)
        R.pending.delete(m), R.done.delete(m);
    if ((m.f & nn) !== 0 && (Br(m), s && (m.nodes?.a?.unfix(), (_ ??= /* @__PURE__ */ new Set()).delete(m))), (m.f & lt) !== 0)
      if (m.f ^= lt, m === h)
        ir(m, null, r);
      else {
        var E = d ? d.next : h;
        m === e.effect.last && (e.effect.last = m.prev), m.prev && (m.prev.next = m.next), m.next && (m.next.prev = m.prev), _t(e, d, m), _t(e, m, E), ir(m, E, r), d = m, b = [], H = [], h = rr(d.next);
        continue;
      }
    if (m !== h) {
      if (c !== void 0 && c.has(m)) {
        if (b.length < H.length) {
          var T = H[0], y;
          d = T.prev;
          var S = b[0], B = b[b.length - 1];
          for (y = 0; y < b.length; y += 1)
            ir(b[y], T, r);
          for (y = 0; y < H.length; y += 1)
            c.delete(H[y]);
          _t(e, S.prev, B.next), _t(e, d, S), _t(e, B, T), h = T, d = B, v -= 1, b = [], H = [];
        } else
          c.delete(m), ir(m, h, r), _t(e, m.prev, m.next), _t(e, m, d === null ? e.effect.first : d.next), _t(e, d, m), d = m;
        continue;
      }
      for (b = [], H = []; h !== null && h !== m; )
        (c ??= /* @__PURE__ */ new Set()).add(h), H.push(h), h = rr(h.next);
      if (h === null)
        continue;
    }
    (m.f & lt) === 0 && b.push(m), d = m, h = rr(m.next);
  }
  if (e.outrogroups !== null) {
    for (const R of e.outrogroups)
      R.pending.size === 0 && (En(e, Dn(R.done)), e.outrogroups?.delete(R));
    e.outrogroups.size === 0 && (e.outrogroups = null);
  }
  if (h !== null || c !== void 0) {
    var N = [];
    if (c !== void 0)
      for (m of c)
        (m.f & nn) === 0 && N.push(m);
    for (; h !== null; )
      (h.f & nn) === 0 && h !== e.fallback && N.push(h), h = rr(h.next);
    var k = N.length;
    if (k > 0) {
      var U = (n & Xi) !== 0 && u === 0 ? r : null;
      if (s) {
        for (v = 0; v < k; v += 1)
          N[v].nodes?.a?.measure();
        for (v = 0; v < k; v += 1)
          N[v].nodes?.a?.fix();
      }
      $s(e, N, U);
    }
  }
  s && ki(() => {
    if (_ !== void 0)
      for (m of _)
        m.nodes?.a?.apply();
  });
}
function to(e, t, r, n, i, s, u, f) {
  var h = (u & is) !== 0 ? (u & as) === 0 ? Mt(r, !1, !1) : si(r) : null, c = (u & ns) !== 0 ? si(i) : null;
  return {
    v: h,
    i: c,
    e: Ot(() => (s(t, h ?? r, c ?? i, f), () => {
      e.delete(n);
    }))
  };
}
function ir(e, t, r) {
  if (e.nodes)
    for (var n = e.nodes.start, i = e.nodes.end, s = t && (t.f & lt) === 0 ? (
      /** @type {EffectNodes} */
      t.nodes.start
    ) : r; n !== null; ) {
      var u = (
        /** @type {TemplateNode} */
        ls(n)
      );
      if (s.before(n), n === i)
        return;
      n = u;
    }
}
function _t(e, t, r) {
  t === null ? e.effect.first = r : t.next = r, r === null ? e.effect.last = t : r.prev = t;
}
function Tn(e, t, r, n, i) {
  var s = t.$$slots?.[r], u = !1;
  s === !0 && (s = t[r === "default" ? "children" : r], u = !0), s === void 0 || s(e, u ? () => n : n);
}
function ro(e, t, r) {
  var n = new Vr(e);
  ur(() => {
    var i = t() ?? null;
    n.ensure(i, i && ((s) => r(s, i)));
  }, Gr);
}
const no = () => performance.now(), et = {
  // don't access requestAnimationFrame eagerly outside method
  // this allows basic testing of user code without JSDOM
  // bunder will eval and remove ternary when the user's app is built
  tick: (
    /** @param {any} _ */
    (e) => requestAnimationFrame(e)
  ),
  now: () => no(),
  tasks: /* @__PURE__ */ new Set()
};
function Ki() {
  const e = et.now();
  et.tasks.forEach((t) => {
    t.c(e) || (et.tasks.delete(t), t.f());
  }), et.tasks.size !== 0 && et.tick(Ki);
}
function io(e) {
  let t;
  return et.tasks.size === 0 && et.tick(Ki), {
    promise: new Promise((r) => {
      et.tasks.add(t = { c: e, f: r });
    }),
    abort() {
      et.tasks.delete(t);
    }
  };
}
function ao(e, t, r, n, i, s) {
  var u = null, f = (
    /** @type {TemplateNode} */
    e
  ), h = new Vr(f, !1);
  ur(() => {
    const c = t() || null;
    var d = c === "svg" ? fs : void 0;
    if (c === null) {
      h.ensure(null, null);
      return;
    }
    return h.ensure(c, (_) => {
      if (c) {
        if (u = Di(c, d), Vt(u, u), n) {
          var b = null, H = u.appendChild(Et());
          n(u, H), b?.remove();
        }
        Dr.nodes.end = u, _.before(u);
      }
    }), () => {
    };
  }, Gr), Rn(() => {
  });
}
function so(e, t) {
  var r = void 0, n;
  Wi(() => {
    r !== (r = t()) && (n && (Pt(n), n = null), r && (n = Ot(() => {
      Gn(() => (
        /** @type {(node: Element) => void} */
        r(e)
      ));
    })));
  });
}
function $i(e) {
  var t, r, n = "";
  if (typeof e == "string" || typeof e == "number") n += e;
  else if (typeof e == "object") if (Array.isArray(e)) {
    var i = e.length;
    for (t = 0; t < i; t++) e[t] && (r = $i(e[t])) && (n && (n += " "), n += r);
  } else for (r in e) e[r] && (n && (n += " "), n += r);
  return n;
}
function oo() {
  for (var e, t, r = 0, n = "", i = arguments.length; r < i; r++) (e = arguments[r]) && (t = $i(e)) && (n && (n += " "), n += t);
  return n;
}
function lo(e) {
  return typeof e == "object" ? oo(e) : e ?? "";
}
const hi = [...` 	
\r\f \v\uFEFF`];
function uo(e, t, r) {
  var n = e == null ? "" : "" + e;
  if (t && (n = n ? n + " " + t : t), r) {
    for (var i of Object.keys(r))
      if (r[i])
        n = n ? n + " " + i : i;
      else if (n.length)
        for (var s = i.length, u = 0; (u = n.indexOf(i, u)) >= 0; ) {
          var f = u + s;
          (u === 0 || hi.includes(n[u - 1])) && (f === n.length || hi.includes(n[f])) ? n = (u === 0 ? "" : n.substring(0, u)) + n.substring(f + 1) : u = f;
        }
  }
  return n === "" ? null : n;
}
function di(e, t = !1) {
  var r = t ? " !important;" : ";", n = "";
  for (var i of Object.keys(e)) {
    var s = e[i];
    s != null && s !== "" && (n += " " + i + ": " + s + r);
  }
  return n;
}
function an(e) {
  return e[0] !== "-" || e[1] !== "-" ? e.toLowerCase() : e;
}
function fo(e, t) {
  if (t) {
    var r = "", n, i;
    if (Array.isArray(t) ? (n = t[0], i = t[1]) : n = t, e) {
      e = String(e).replaceAll(/\s*\/\*.*?\*\/\s*/g, "").trim();
      var s = !1, u = 0, f = !1, h = [];
      n && h.push(...Object.keys(n).map(an)), i && h.push(...Object.keys(i).map(an));
      var c = 0, d = -1;
      const w = e.length;
      for (var _ = 0; _ < w; _++) {
        var b = e[_];
        if (f ? b === "/" && e[_ - 1] === "*" && (f = !1) : s ? s === b && (s = !1) : b === "/" && e[_ + 1] === "*" ? f = !0 : b === '"' || b === "'" ? s = b : b === "(" ? u++ : b === ")" && u--, !f && s === !1 && u === 0) {
          if (b === ":" && d === -1)
            d = _;
          else if (b === ";" || _ === w - 1) {
            if (d !== -1) {
              var H = an(e.substring(c, d).trim());
              if (!h.includes(H)) {
                b !== ";" && _++;
                var O = e.substring(c, _).trim();
                r += " " + O + ";";
              }
            }
            c = _ + 1, d = -1;
          }
        }
      }
    }
    return n && (r += di(n)), i && (r += di(i, !0)), r = r.trim(), r === "" ? null : r;
  }
  return e == null ? null : String(e);
}
function ht(e, t, r, n, i, s) {
  var u = (
    /** @type {any} */
    e[oi]
  );
  if (u !== r || u === void 0) {
    var f = uo(r, n, s);
    f == null ? e.removeAttribute("class") : t ? e.className = f : e.setAttribute("class", f), e[oi] = r;
  } else if (s && i !== s)
    for (var h in s) {
      var c = !!s[h];
      (i == null || c !== !!i[h]) && e.classList.toggle(h, c);
    }
  return s;
}
function sn(e, t = {}, r, n) {
  for (var i in r) {
    var s = r[i];
    t[i] !== s && (r[i] == null ? e.style.removeProperty(i) : e.style.setProperty(i, s, n));
  }
}
function qe(e, t, r, n) {
  var i = (
    /** @type {any} */
    e[li]
  );
  if (i !== t) {
    var s = fo(t, n);
    s == null ? e.removeAttribute("style") : e.style.cssText = s, e[li] = t;
  } else n && (Array.isArray(n) ? (sn(e, r?.[0], n[0]), sn(e, r?.[1], n[1], "important")) : sn(e, r, n));
  return n;
}
function Cr(e, t, r = !1) {
  if (e.multiple) {
    if (t == null)
      return;
    if (!Cn(t))
      return cs();
    for (var n of e.options)
      n.selected = t.includes(pi(n));
    return;
  }
  for (n of e.options) {
    var i = pi(n);
    if (hs(i, t)) {
      n.selected = !0;
      return;
    }
  }
  (!r || t !== void 0) && (e.selectedIndex = -1);
}
function ea(e) {
  var t = new MutationObserver(() => {
    Cr(e, e.__value);
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
  }), Rn(() => {
    t.disconnect();
  });
}
function pi(e) {
  return "__value" in e ? e.__value : e.value;
}
const ar = /* @__PURE__ */ Symbol("class"), Ft = /* @__PURE__ */ Symbol("style"), ta = /* @__PURE__ */ Symbol("is custom element"), ra = /* @__PURE__ */ Symbol("is html"), co = Fn ? "input" : "INPUT", ho = Fn ? "option" : "OPTION", po = Fn ? "select" : "SELECT";
function vo(e, t) {
  t ? e.hasAttribute("selected") || e.setAttribute("selected", "") : e.removeAttribute("selected");
}
function wt(e, t, r, n) {
  var i = na(e);
  i[t] !== (i[t] = r) && (t === "loading" && (e[ds] = r), r == null ? e.removeAttribute(t) : typeof r != "string" && ia(e).includes(t) ? e[t] = r : e.setAttribute(t, r));
}
function mo(e, t, r, n, i = !1, s = !1) {
  var u = na(e), f = u[ta], h = !u[ra], c = t || {}, d = e.nodeName === ho;
  for (var _ in t)
    _ in r || (r[_] = null);
  r.class ? r.class = lo(r.class) : r.class = null, r[Ft] && (r.style ??= null);
  var b = ia(e);
  if (e.nodeName === co && "type" in r && ("value" in r || "__value" in r)) {
    var H = r.type;
    (H !== c.type || H === void 0 && e.hasAttribute("type")) && (c.type = H, wt(e, "type", H));
  }
  for (const y in r) {
    let S = r[y];
    if (d && y === "value" && S == null) {
      e.value = e.__value = "", c[y] = S;
      continue;
    }
    if (y === "class") {
      var O = e.namespaceURI === "http://www.w3.org/1999/xhtml";
      ht(e, O, S, n, t?.[ar], r[ar]), c[y] = S, c[ar] = r[ar];
      continue;
    }
    if (y === "style") {
      qe(e, S, t?.[Ft], r[Ft]), c[y] = S, c[Ft] = r[Ft];
      continue;
    }
    var w = c[y];
    if (!(S === w && !(S === void 0 && e.hasAttribute(y)))) {
      c[y] = S;
      var m = y[0] + y[1];
      if (m !== "$$")
        if (m === "on") {
          const B = {}, N = "$$" + y;
          let k = y.slice(2);
          var v = ws(k);
          if (_s(k) && (k = k.slice(0, -7), B.capture = !0), !v && w) {
            if (S != null) continue;
            e.removeEventListener(k, c[N], B), c[N] = null;
          }
          if (v)
            qi(k, e, S), Fr([k]);
          else if (S != null) {
            let U = function(R) {
              c[y].call(this, R);
            };
            c[N] = ys(k, e, U, B);
          }
        } else if (y === "style")
          wt(e, y, S);
        else if (y === "autofocus")
          Xs(
            /** @type {HTMLElement} */
            e,
            !!S
          );
        else if (!f && (y === "__value" || y === "value" && S != null))
          e.value = e.__value = S;
        else if (y === "selected" && d)
          vo(
            /** @type {HTMLOptionElement} */
            e,
            S
          );
        else {
          var E = y;
          h || (E = xs(E));
          var T = E === "defaultValue" || E === "defaultChecked";
          if (S == null && !f && !T)
            if (u[y] = null, E === "value" || E === "checked") {
              let B = (
                /** @type {HTMLInputElement} */
                e
              );
              const N = t === void 0;
              if (E === "value") {
                let k = B.defaultValue;
                B.removeAttribute(E), B.defaultValue = k, B.value = B.__value = N ? k : null;
              } else {
                let k = B.defaultChecked;
                B.removeAttribute(E), B.defaultChecked = k, B.checked = N ? k : !1;
              }
            } else
              e.removeAttribute(y);
          else T || b.includes(E) && (f || typeof S != "string") ? (e[E] = S, E in u && (u[E] = Es)) : typeof S != "function" && wt(e, E, S);
        }
    }
  }
  return c;
}
function go(e, t, r = [], n = [], i = [], s, u = !1, f = !1) {
  gs(i, r, n, (h) => {
    var c = void 0, d = {}, _ = e.nodeName === po, b = !1;
    if (Wi(() => {
      var O = t(...h.map(o)), w = mo(
        e,
        c,
        O,
        s,
        u,
        f
      );
      b && _ && "value" in O && Cr(
        /** @type {HTMLSelectElement} */
        e,
        O.value
      );
      for (let v of Object.getOwnPropertySymbols(d))
        O[v] || Pt(d[v]);
      for (let v of Object.getOwnPropertySymbols(O)) {
        var m = O[v];
        v.description === bs && (!c || m !== c[v]) && (d[v] && Pt(d[v]), d[v] = Ot(() => so(e, () => m))), w[v] = m;
      }
      c = w;
    }), _) {
      var H = (
        /** @type {HTMLSelectElement} */
        e
      );
      Gn(() => {
        Cr(
          H,
          /** @type {Record<string | symbol, any>} */
          c.value,
          !0
        ), ea(H);
      });
    }
    b = !0;
  });
}
function na(e) {
  return (
    /** @type {Record<string | symbol, unknown>} **/
    /** @type {any} */
    e[ps] ??= {
      [ta]: e.nodeName.includes("-"),
      [ra]: e.namespaceURI === vs
    }
  );
}
var vi = /* @__PURE__ */ new Map();
function ia(e) {
  var t = e.getAttribute("is") || e.nodeName, r = vi.get(t);
  if (r) return r;
  vi.set(t, r = []);
  for (var n, i = e, s = Element.prototype; s !== i; ) {
    n = ms(i);
    for (var u in n)
      n[u].set && // better safe than sorry, we don't want spread attributes to mess with HTML content
      u !== "innerHTML" && u !== "textContent" && u !== "innerText" && r.push(u);
    i = Ri(i);
  }
  return r;
}
function on(e, t) {
  return e === t || e?.[Un] === t;
}
function sr(e = {}, t, r, n) {
  var i = (
    /** @type {ComponentContext} */
    jt.r
  ), s = (
    /** @type {Effect} */
    Dr
  );
  return Gn(() => {
    var u, f;
    return Ts(() => {
      u = f, f = [], xe(() => {
        on(r(...f), e) || (t(e, ...f), u && on(r(...u), e) && t(null, ...u));
      });
    }), () => {
      let h = s;
      for (; h !== i && h.parent !== null && h.parent.f & Ss; )
        h = h.parent;
      const c = () => {
        f && on(r(...f), e) && t(null, ...f);
      }, d = h.teardown;
      h.teardown = () => {
        c(), d?.();
      };
    };
  }), e;
}
function bo(e = !1) {
  const t = (
    /** @type {ComponentContextLegacy} */
    jt
  ), r = t.l.u;
  if (!r) return;
  let n = () => ze(t.s);
  if (e) {
    let i = 0, s = (
      /** @type {Record<string, any>} */
      {}
    );
    const u = bn(() => {
      let f = !1;
      const h = t.s;
      for (const c in h)
        h[c] !== s[c] && (s[c] = h[c], f = !0);
      return f && i++, i;
    });
    n = () => o(u);
  }
  r.b.length && As(() => {
    mi(t, n), gn(r.b);
  }), We(() => {
    const i = xe(() => r.m.map(Hs));
    return () => {
      for (const s of i)
        typeof s == "function" && s();
    };
  }), r.a.length && We(() => {
    mi(t, n), gn(r.a);
  });
}
function mi(e, t) {
  if (e.l.s)
    for (const r of e.l.s) o(r);
  t();
}
const _o = {
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
function yo(e, t, r) {
  return new Proxy(
    { props: e, exclude: t },
    _o
  );
}
const xo = {
  get(e, t) {
    let r = e.props.length;
    for (; r--; ) {
      let n = e.props[r];
      if (tr(n) && (n = n()), typeof n == "object" && n !== null && t in n) return n[t];
    }
  },
  set(e, t, r) {
    let n = e.props.length;
    for (; n--; ) {
      let i = e.props[n];
      tr(i) && (i = i());
      const s = _n(i, t);
      if (s && s.set)
        return s.set(r), !0;
    }
    return !1;
  },
  getOwnPropertyDescriptor(e, t) {
    let r = e.props.length;
    for (; r--; ) {
      let n = e.props[r];
      if (tr(n) && (n = n()), typeof n == "object" && n !== null && t in n) {
        const i = _n(n, t);
        return i && !i.configurable && (i.configurable = !0), i;
      }
    }
  },
  has(e, t) {
    if (t === Un || t === Zi) return !1;
    for (let r of e.props)
      if (tr(r) && (r = r()), r != null && t in r) return !0;
    return !1;
  },
  ownKeys(e) {
    const t = [];
    for (let r of e.props)
      if (tr(r) && (r = r()), !!r) {
        for (const n in r)
          t.includes(n) || t.push(n);
        for (const n of Object.getOwnPropertySymbols(r))
          t.includes(n) || t.push(n);
      }
    return t;
  }
};
function Eo(...e) {
  return new Proxy({ props: e }, xo);
}
function C(e, t, r, n) {
  var i = !ji || (r & Ns) !== 0, s = (r & Os) !== 0, u = (r & Ls) !== 0, f = (
    /** @type {V} */
    n
  ), h = !0, c = (
    /** @type {Derived<V> | undefined} */
    void 0
  ), d = () => u && i ? (c ??= bn(
    /** @type {() => V} */
    n
  ), o(c)) : (h && (h = !1, f = u ? xe(
    /** @type {() => V} */
    n
  ) : (
    /** @type {V} */
    n
  )), f);
  let _;
  if (s) {
    var b = Un in e || Zi in e;
    _ = _n(e, t)?.set ?? (b && t in e ? (y) => e[t] = y : void 0);
  }
  var H, O = !1;
  s ? [H, O] = zs(() => (
    /** @type {V} */
    e[t]
  )) : H = /** @type {V} */
  e[t], H === void 0 && n !== void 0 && (H = d(), _ && (i && Ms(), _(H)));
  var w;
  if (i ? w = () => {
    var y = (
      /** @type {V} */
      e[t]
    );
    return y === void 0 ? d() : (h = !0, y);
  } : w = () => {
    var y = (
      /** @type {V} */
      e[t]
    );
    return y !== void 0 && (f = /** @type {V} */
    void 0), y === void 0 ? f : y;
  }, i && (r & Ps) === 0)
    return w;
  if (_) {
    var m = e.$$legacy;
    return (
      /** @type {() => V} */
      (function(y, S) {
        return arguments.length > 0 ? ((!i || !S || m || O) && _(S ? w() : y), y) : w();
      })
    );
  }
  var v = !1, E = ((r & Is) !== 0 ? bn : Vi)(() => (v = !1, w()));
  s && o(E);
  var T = (
    /** @type {Effect} */
    Dr
  );
  return (
    /** @type {() => V} */
    (function(y, S) {
      if (arguments.length > 0) {
        const B = S ? o(E) : i && s ? or(y) : y;
        return x(E, B), v = !0, f !== void 0 && (f = B), y;
      }
      return Bs && v || (T.f & zi) !== 0 ? E.v : o(E);
    })
  );
}
const wo = [
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
], gi = {
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
wo.reduce((e, { color: t, primary: r, secondary: n }) => ({
  ...e,
  [t]: {
    primary: gi[t][r],
    secondary: gi[t][n]
  }
}), {});
function To(e) {
  return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, "default") ? e.default : e;
}
var ln, bi;
function So() {
  if (bi) return ln;
  bi = 1;
  var e = function(E) {
    return t(E) && !r(E);
  };
  function t(v) {
    return !!v && typeof v == "object";
  }
  function r(v) {
    var E = Object.prototype.toString.call(v);
    return E === "[object RegExp]" || E === "[object Date]" || s(v);
  }
  var n = typeof Symbol == "function" && Symbol.for, i = n ? /* @__PURE__ */ Symbol.for("react.element") : 60103;
  function s(v) {
    return v.$$typeof === i;
  }
  function u(v) {
    return Array.isArray(v) ? [] : {};
  }
  function f(v, E) {
    return E.clone !== !1 && E.isMergeableObject(v) ? w(u(v), v, E) : v;
  }
  function h(v, E, T) {
    return v.concat(E).map(function(y) {
      return f(y, T);
    });
  }
  function c(v, E) {
    if (!E.customMerge)
      return w;
    var T = E.customMerge(v);
    return typeof T == "function" ? T : w;
  }
  function d(v) {
    return Object.getOwnPropertySymbols ? Object.getOwnPropertySymbols(v).filter(function(E) {
      return Object.propertyIsEnumerable.call(v, E);
    }) : [];
  }
  function _(v) {
    return Object.keys(v).concat(d(v));
  }
  function b(v, E) {
    try {
      return E in v;
    } catch {
      return !1;
    }
  }
  function H(v, E) {
    return b(v, E) && !(Object.hasOwnProperty.call(v, E) && Object.propertyIsEnumerable.call(v, E));
  }
  function O(v, E, T) {
    var y = {};
    return T.isMergeableObject(v) && _(v).forEach(function(S) {
      y[S] = f(v[S], T);
    }), _(E).forEach(function(S) {
      H(v, S) || (b(v, S) && T.isMergeableObject(E[S]) ? y[S] = c(S, T)(v[S], E[S], T) : y[S] = f(E[S], T));
    }), y;
  }
  function w(v, E, T) {
    T = T || {}, T.arrayMerge = T.arrayMerge || h, T.isMergeableObject = T.isMergeableObject || e, T.cloneUnlessOtherwiseSpecified = f;
    var y = Array.isArray(E), S = Array.isArray(v), B = y === S;
    return B ? y ? T.arrayMerge(v, E, T) : O(v, E, T) : f(E, T);
  }
  w.all = function(E, T) {
    if (!Array.isArray(E))
      throw new Error("first argument should be an array");
    return E.reduce(function(y, S) {
      return w(y, S, T);
    }, {});
  };
  var m = w;
  return ln = m, ln;
}
var Ao = So();
const Ho = /* @__PURE__ */ To(Ao);
var Sn = function(e, t) {
  return Sn = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(r, n) {
    r.__proto__ = n;
  } || function(r, n) {
    for (var i in n) Object.prototype.hasOwnProperty.call(n, i) && (r[i] = n[i]);
  }, Sn(e, t);
};
function zr(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Class extends value " + String(t) + " is not a constructor or null");
  Sn(e, t);
  function r() {
    this.constructor = e;
  }
  e.prototype = t === null ? Object.create(t) : (r.prototype = t.prototype, new r());
}
var Z = function() {
  return Z = Object.assign || function(t) {
    for (var r, n = 1, i = arguments.length; n < i; n++) {
      r = arguments[n];
      for (var s in r) Object.prototype.hasOwnProperty.call(r, s) && (t[s] = r[s]);
    }
    return t;
  }, Z.apply(this, arguments);
};
function Mo(e, t) {
  var r = {};
  for (var n in e) Object.prototype.hasOwnProperty.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && typeof Object.getOwnPropertySymbols == "function")
    for (var i = 0, n = Object.getOwnPropertySymbols(e); i < n.length; i++)
      t.indexOf(n[i]) < 0 && Object.prototype.propertyIsEnumerable.call(e, n[i]) && (r[n[i]] = e[n[i]]);
  return r;
}
function un(e, t, r) {
  if (r || arguments.length === 2) for (var n = 0, i = t.length, s; n < i; n++)
    (s || !(n in t)) && (s || (s = Array.prototype.slice.call(t, 0, n)), s[n] = t[n]);
  return e.concat(s || Array.prototype.slice.call(t));
}
function fn(e, t) {
  var r = t && t.cache ? t.cache : Co, n = t && t.serializer ? t.serializer : Lo, i = t && t.strategy ? t.strategy : No;
  return i(e, {
    cache: r,
    serializer: n
  });
}
function Po(e) {
  return e == null || typeof e == "number" || typeof e == "boolean";
}
function Oo(e, t, r, n) {
  var i = Po(n) ? n : r(n), s = t.get(i);
  return typeof s > "u" && (s = e.call(this, n), t.set(i, s)), s;
}
function aa(e, t, r) {
  var n = Array.prototype.slice.call(arguments, 3), i = r(n), s = t.get(i);
  return typeof s > "u" && (s = e.apply(this, n), t.set(i, s)), s;
}
function sa(e, t, r, n, i) {
  return r.bind(t, e, n, i);
}
function No(e, t) {
  var r = e.length === 1 ? Oo : aa;
  return sa(e, this, r, t.cache.create(), t.serializer);
}
function Io(e, t) {
  return sa(e, this, aa, t.cache.create(), t.serializer);
}
var Lo = function() {
  return JSON.stringify(arguments);
}, Bo = (
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
), Co = {
  create: function() {
    return new Bo();
  }
}, cn = {
  variadic: Io
}, V;
(function(e) {
  e[e.EXPECT_ARGUMENT_CLOSING_BRACE = 1] = "EXPECT_ARGUMENT_CLOSING_BRACE", e[e.EMPTY_ARGUMENT = 2] = "EMPTY_ARGUMENT", e[e.MALFORMED_ARGUMENT = 3] = "MALFORMED_ARGUMENT", e[e.EXPECT_ARGUMENT_TYPE = 4] = "EXPECT_ARGUMENT_TYPE", e[e.INVALID_ARGUMENT_TYPE = 5] = "INVALID_ARGUMENT_TYPE", e[e.EXPECT_ARGUMENT_STYLE = 6] = "EXPECT_ARGUMENT_STYLE", e[e.INVALID_NUMBER_SKELETON = 7] = "INVALID_NUMBER_SKELETON", e[e.INVALID_DATE_TIME_SKELETON = 8] = "INVALID_DATE_TIME_SKELETON", e[e.EXPECT_NUMBER_SKELETON = 9] = "EXPECT_NUMBER_SKELETON", e[e.EXPECT_DATE_TIME_SKELETON = 10] = "EXPECT_DATE_TIME_SKELETON", e[e.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE = 11] = "UNCLOSED_QUOTE_IN_ARGUMENT_STYLE", e[e.EXPECT_SELECT_ARGUMENT_OPTIONS = 12] = "EXPECT_SELECT_ARGUMENT_OPTIONS", e[e.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE = 13] = "EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE = 14] = "INVALID_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR = 15] = "EXPECT_SELECT_ARGUMENT_SELECTOR", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR = 16] = "EXPECT_PLURAL_ARGUMENT_SELECTOR", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT = 17] = "EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT = 18] = "EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT", e[e.INVALID_PLURAL_ARGUMENT_SELECTOR = 19] = "INVALID_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_PLURAL_ARGUMENT_SELECTOR = 20] = "DUPLICATE_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_SELECT_ARGUMENT_SELECTOR = 21] = "DUPLICATE_SELECT_ARGUMENT_SELECTOR", e[e.MISSING_OTHER_CLAUSE = 22] = "MISSING_OTHER_CLAUSE", e[e.INVALID_TAG = 23] = "INVALID_TAG", e[e.INVALID_TAG_NAME = 25] = "INVALID_TAG_NAME", e[e.UNMATCHED_CLOSING_TAG = 26] = "UNMATCHED_CLOSING_TAG", e[e.UNCLOSED_TAG = 27] = "UNCLOSED_TAG";
})(V || (V = {}));
var ie;
(function(e) {
  e[e.literal = 0] = "literal", e[e.argument = 1] = "argument", e[e.number = 2] = "number", e[e.date = 3] = "date", e[e.time = 4] = "time", e[e.select = 5] = "select", e[e.plural = 6] = "plural", e[e.pound = 7] = "pound", e[e.tag = 8] = "tag";
})(ie || (ie = {}));
var zt;
(function(e) {
  e[e.number = 0] = "number", e[e.dateTime = 1] = "dateTime";
})(zt || (zt = {}));
function _i(e) {
  return e.type === ie.literal;
}
function Ro(e) {
  return e.type === ie.argument;
}
function oa(e) {
  return e.type === ie.number;
}
function la(e) {
  return e.type === ie.date;
}
function ua(e) {
  return e.type === ie.time;
}
function fa(e) {
  return e.type === ie.select;
}
function ca(e) {
  return e.type === ie.plural;
}
function ko(e) {
  return e.type === ie.pound;
}
function ha(e) {
  return e.type === ie.tag;
}
function da(e) {
  return !!(e && typeof e == "object" && e.type === zt.number);
}
function An(e) {
  return !!(e && typeof e == "object" && e.type === zt.dateTime);
}
var pa = /[ \xA0\u1680\u2000-\u200A\u202F\u205F\u3000]/, Do = /(?:[Eec]{1,6}|G{1,5}|[Qq]{1,5}|(?:[yYur]+|U{1,5})|[ML]{1,5}|d{1,2}|D{1,3}|F{1}|[abB]{1,5}|[hkHK]{1,2}|w{1,2}|W{1}|m{1,2}|s{1,2}|[zZOvVxX]{1,4})(?=([^']*'[^']*')*[^']*$)/g;
function Go(e) {
  var t = {};
  return e.replace(Do, function(r) {
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
var Fo = /[\t-\r \x85\u200E\u200F\u2028\u2029]/i;
function Uo(e) {
  if (e.length === 0)
    throw new Error("Number skeleton cannot be empty");
  for (var t = e.split(Fo).filter(function(b) {
    return b.length > 0;
  }), r = [], n = 0, i = t; n < i.length; n++) {
    var s = i[n], u = s.split("/");
    if (u.length === 0)
      throw new Error("Invalid number skeleton");
    for (var f = u[0], h = u.slice(1), c = 0, d = h; c < d.length; c++) {
      var _ = d[c];
      if (_.length === 0)
        throw new Error("Invalid number skeleton");
    }
    r.push({ stem: f, options: h });
  }
  return r;
}
function jo(e) {
  return e.replace(/^(.*?)-/, "");
}
var yi = /^\.(?:(0+)(\*)?|(#+)|(0+)(#+))$/g, va = /^(@+)?(\+|#+)?[rs]?$/g, Vo = /(\*)(0+)|(#+)(0+)|(0+)/g, ma = /^(0+)$/;
function xi(e) {
  var t = {};
  return e[e.length - 1] === "r" ? t.roundingPriority = "morePrecision" : e[e.length - 1] === "s" && (t.roundingPriority = "lessPrecision"), e.replace(va, function(r, n, i) {
    return typeof i != "string" ? (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length) : i === "+" ? t.minimumSignificantDigits = n.length : n[0] === "#" ? t.maximumSignificantDigits = n.length : (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length + (typeof i == "string" ? i.length : 0)), "";
  }), t;
}
function ga(e) {
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
function zo(e) {
  var t;
  if (e[0] === "E" && e[1] === "E" ? (t = {
    notation: "engineering"
  }, e = e.slice(2)) : e[0] === "E" && (t = {
    notation: "scientific"
  }, e = e.slice(1)), t) {
    var r = e.slice(0, 2);
    if (r === "+!" ? (t.signDisplay = "always", e = e.slice(2)) : r === "+?" && (t.signDisplay = "exceptZero", e = e.slice(2)), !ma.test(e))
      throw new Error("Malformed concise eng/scientific notation");
    t.minimumIntegerDigits = e.length;
  }
  return t;
}
function Ei(e) {
  var t = {}, r = ga(e);
  return r || t;
}
function Xo(e) {
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
        t.style = "unit", t.unit = jo(i.options[0]);
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
        t = Z(Z(Z({}, t), { notation: "scientific" }), i.options.reduce(function(h, c) {
          return Z(Z({}, h), Ei(c));
        }, {}));
        continue;
      case "engineering":
        t = Z(Z(Z({}, t), { notation: "engineering" }), i.options.reduce(function(h, c) {
          return Z(Z({}, h), Ei(c));
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
        i.options[0].replace(Vo, function(h, c, d, _, b, H) {
          if (c)
            t.minimumIntegerDigits = d.length;
          else {
            if (_ && b)
              throw new Error("We currently do not support maximum integer digits");
            if (H)
              throw new Error("We currently do not support exact integer digits");
          }
          return "";
        });
        continue;
    }
    if (ma.test(i.stem)) {
      t.minimumIntegerDigits = i.stem.length;
      continue;
    }
    if (yi.test(i.stem)) {
      if (i.options.length > 1)
        throw new RangeError("Fraction-precision stems only accept a single optional option");
      i.stem.replace(yi, function(h, c, d, _, b, H) {
        return d === "*" ? t.minimumFractionDigits = c.length : _ && _[0] === "#" ? t.maximumFractionDigits = _.length : b && H ? (t.minimumFractionDigits = b.length, t.maximumFractionDigits = b.length + H.length) : (t.minimumFractionDigits = c.length, t.maximumFractionDigits = c.length), "";
      });
      var s = i.options[0];
      s === "w" ? t = Z(Z({}, t), { trailingZeroDisplay: "stripIfInteger" }) : s && (t = Z(Z({}, t), xi(s)));
      continue;
    }
    if (va.test(i.stem)) {
      t = Z(Z({}, t), xi(i.stem));
      continue;
    }
    var u = ga(i.stem);
    u && (t = Z(Z({}, t), u));
    var f = zo(i.stem);
    f && (t = Z(Z({}, t), f));
  }
  return t;
}
var Mr = {
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
function Wo(e, t) {
  for (var r = "", n = 0; n < e.length; n++) {
    var i = e.charAt(n);
    if (i === "j") {
      for (var s = 0; n + 1 < e.length && e.charAt(n + 1) === i; )
        s++, n++;
      var u = 1 + (s & 1), f = s < 2 ? 1 : 3 + (s >> 1), h = "a", c = qo(t);
      for ((c == "H" || c == "k") && (f = 0); f-- > 0; )
        r += h;
      for (; u-- > 0; )
        r = c + r;
    } else i === "J" ? r += "H" : r += i;
  }
  return r;
}
function qo(e) {
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
  var i = Mr[n || ""] || Mr[r || ""] || Mr["".concat(r, "-001")] || Mr["001"];
  return i[0];
}
var hn, Zo = new RegExp("^".concat(pa.source, "*")), Yo = new RegExp("".concat(pa.source, "*$"));
function X(e, t) {
  return { start: e, end: t };
}
var Jo = !!String.prototype.startsWith && "_a".startsWith("a", 1), Qo = !!String.fromCodePoint, Ko = !!Object.fromEntries, $o = !!String.prototype.codePointAt, el = !!String.prototype.trimStart, tl = !!String.prototype.trimEnd, rl = !!Number.isSafeInteger, nl = rl ? Number.isSafeInteger : function(e) {
  return typeof e == "number" && isFinite(e) && Math.floor(e) === e && Math.abs(e) <= 9007199254740991;
}, Hn = !0;
try {
  var il = _a("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  Hn = ((hn = il.exec("a")) === null || hn === void 0 ? void 0 : hn[0]) === "a";
} catch {
  Hn = !1;
}
var wi = Jo ? (
  // Native
  function(t, r, n) {
    return t.startsWith(r, n);
  }
) : (
  // For IE11
  function(t, r, n) {
    return t.slice(n, n + r.length) === r;
  }
), Mn = Qo ? String.fromCodePoint : (
  // IE11
  function() {
    for (var t = [], r = 0; r < arguments.length; r++)
      t[r] = arguments[r];
    for (var n = "", i = t.length, s = 0, u; i > s; ) {
      if (u = t[s++], u > 1114111)
        throw RangeError(u + " is not a valid code point");
      n += u < 65536 ? String.fromCharCode(u) : String.fromCharCode(((u -= 65536) >> 10) + 55296, u % 1024 + 56320);
    }
    return n;
  }
), Ti = (
  // native
  Ko ? Object.fromEntries : (
    // Ponyfill
    function(t) {
      for (var r = {}, n = 0, i = t; n < i.length; n++) {
        var s = i[n], u = s[0], f = s[1];
        r[u] = f;
      }
      return r;
    }
  )
), ba = $o ? (
  // Native
  function(t, r) {
    return t.codePointAt(r);
  }
) : (
  // IE 11
  function(t, r) {
    var n = t.length;
    if (!(r < 0 || r >= n)) {
      var i = t.charCodeAt(r), s;
      return i < 55296 || i > 56319 || r + 1 === n || (s = t.charCodeAt(r + 1)) < 56320 || s > 57343 ? i : (i - 55296 << 10) + (s - 56320) + 65536;
    }
  }
), al = el ? (
  // Native
  function(t) {
    return t.trimStart();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(Zo, "");
  }
), sl = tl ? (
  // Native
  function(t) {
    return t.trimEnd();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(Yo, "");
  }
);
function _a(e, t) {
  return new RegExp(e, t);
}
var Pn;
if (Hn) {
  var Si = _a("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  Pn = function(t, r) {
    var n;
    Si.lastIndex = r;
    var i = Si.exec(t);
    return (n = i[1]) !== null && n !== void 0 ? n : "";
  };
} else
  Pn = function(t, r) {
    for (var n = []; ; ) {
      var i = ba(t, r);
      if (i === void 0 || ya(i) || fl(i))
        break;
      n.push(i), r += i >= 65536 ? 2 : 1;
    }
    return Mn.apply(void 0, n);
  };
var ol = (
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
        var s = this.char();
        if (s === 123) {
          var u = this.parseArgument(t, n);
          if (u.err)
            return u;
          i.push(u.val);
        } else {
          if (s === 125 && t > 0)
            break;
          if (s === 35 && (r === "plural" || r === "selectordinal")) {
            var f = this.clonePosition();
            this.bump(), i.push({
              type: ie.pound,
              location: X(f, this.clonePosition())
            });
          } else if (s === 60 && !this.ignoreTag && this.peek() === 47) {
            if (n)
              break;
            return this.error(V.UNMATCHED_CLOSING_TAG, X(this.clonePosition(), this.clonePosition()));
          } else if (s === 60 && !this.ignoreTag && On(this.peek() || 0)) {
            var u = this.parseTag(t, r);
            if (u.err)
              return u;
            i.push(u.val);
          } else {
            var u = this.parseLiteral(t, r);
            if (u.err)
              return u;
            i.push(u.val);
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
            type: ie.literal,
            value: "<".concat(i, "/>"),
            location: X(n, this.clonePosition())
          },
          err: null
        };
      if (this.bumpIf(">")) {
        var s = this.parseMessage(t + 1, r, !0);
        if (s.err)
          return s;
        var u = s.val, f = this.clonePosition();
        if (this.bumpIf("</")) {
          if (this.isEOF() || !On(this.char()))
            return this.error(V.INVALID_TAG, X(f, this.clonePosition()));
          var h = this.clonePosition(), c = this.parseTagName();
          return i !== c ? this.error(V.UNMATCHED_CLOSING_TAG, X(h, this.clonePosition())) : (this.bumpSpace(), this.bumpIf(">") ? {
            val: {
              type: ie.tag,
              value: i,
              children: u,
              location: X(n, this.clonePosition())
            },
            err: null
          } : this.error(V.INVALID_TAG, X(f, this.clonePosition())));
        } else
          return this.error(V.UNCLOSED_TAG, X(n, this.clonePosition()));
      } else
        return this.error(V.INVALID_TAG, X(n, this.clonePosition()));
    }, e.prototype.parseTagName = function() {
      var t = this.offset();
      for (this.bump(); !this.isEOF() && ul(this.char()); )
        this.bump();
      return this.message.slice(t, this.offset());
    }, e.prototype.parseLiteral = function(t, r) {
      for (var n = this.clonePosition(), i = ""; ; ) {
        var s = this.tryParseQuote(r);
        if (s) {
          i += s;
          continue;
        }
        var u = this.tryParseUnquoted(t, r);
        if (u) {
          i += u;
          continue;
        }
        var f = this.tryParseLeftAngleBracket();
        if (f) {
          i += f;
          continue;
        }
        break;
      }
      var h = X(n, this.clonePosition());
      return {
        val: { type: ie.literal, value: i, location: h },
        err: null
      };
    }, e.prototype.tryParseLeftAngleBracket = function() {
      return !this.isEOF() && this.char() === 60 && (this.ignoreTag || // If at the opening tag or closing tag position, bail.
      !ll(this.peek() || 0)) ? (this.bump(), "<") : null;
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
      return Mn.apply(void 0, r);
    }, e.prototype.tryParseUnquoted = function(t, r) {
      if (this.isEOF())
        return null;
      var n = this.char();
      return n === 60 || n === 123 || n === 35 && (r === "plural" || r === "selectordinal") || n === 125 && t > 0 ? null : (this.bump(), Mn(n));
    }, e.prototype.parseArgument = function(t, r) {
      var n = this.clonePosition();
      if (this.bump(), this.bumpSpace(), this.isEOF())
        return this.error(V.EXPECT_ARGUMENT_CLOSING_BRACE, X(n, this.clonePosition()));
      if (this.char() === 125)
        return this.bump(), this.error(V.EMPTY_ARGUMENT, X(n, this.clonePosition()));
      var i = this.parseIdentifierIfPossible().value;
      if (!i)
        return this.error(V.MALFORMED_ARGUMENT, X(n, this.clonePosition()));
      if (this.bumpSpace(), this.isEOF())
        return this.error(V.EXPECT_ARGUMENT_CLOSING_BRACE, X(n, this.clonePosition()));
      switch (this.char()) {
        // Simple argument: `{name}`
        case 125:
          return this.bump(), {
            val: {
              type: ie.argument,
              // value does not include the opening and closing braces.
              value: i,
              location: X(n, this.clonePosition())
            },
            err: null
          };
        // Argument with options: `{name, format, ...}`
        case 44:
          return this.bump(), this.bumpSpace(), this.isEOF() ? this.error(V.EXPECT_ARGUMENT_CLOSING_BRACE, X(n, this.clonePosition())) : this.parseArgumentOptions(t, r, i, n);
        default:
          return this.error(V.MALFORMED_ARGUMENT, X(n, this.clonePosition()));
      }
    }, e.prototype.parseIdentifierIfPossible = function() {
      var t = this.clonePosition(), r = this.offset(), n = Pn(this.message, r), i = r + n.length;
      this.bumpTo(i);
      var s = this.clonePosition(), u = X(t, s);
      return { value: n, location: u };
    }, e.prototype.parseArgumentOptions = function(t, r, n, i) {
      var s, u = this.clonePosition(), f = this.parseIdentifierIfPossible().value, h = this.clonePosition();
      switch (f) {
        case "":
          return this.error(V.EXPECT_ARGUMENT_TYPE, X(u, h));
        case "number":
        case "date":
        case "time": {
          this.bumpSpace();
          var c = null;
          if (this.bumpIf(",")) {
            this.bumpSpace();
            var d = this.clonePosition(), _ = this.parseSimpleArgStyleIfPossible();
            if (_.err)
              return _;
            var b = sl(_.val);
            if (b.length === 0)
              return this.error(V.EXPECT_ARGUMENT_STYLE, X(this.clonePosition(), this.clonePosition()));
            var H = X(d, this.clonePosition());
            c = { style: b, styleLocation: H };
          }
          var O = this.tryParseArgumentClose(i);
          if (O.err)
            return O;
          var w = X(i, this.clonePosition());
          if (c && wi(c?.style, "::", 0)) {
            var m = al(c.style.slice(2));
            if (f === "number") {
              var _ = this.parseNumberSkeletonFromString(m, c.styleLocation);
              return _.err ? _ : {
                val: { type: ie.number, value: n, location: w, style: _.val },
                err: null
              };
            } else {
              if (m.length === 0)
                return this.error(V.EXPECT_DATE_TIME_SKELETON, w);
              var v = m;
              this.locale && (v = Wo(m, this.locale));
              var b = {
                type: zt.dateTime,
                pattern: v,
                location: c.styleLocation,
                parsedOptions: this.shouldParseSkeletons ? Go(v) : {}
              }, E = f === "date" ? ie.date : ie.time;
              return {
                val: { type: E, value: n, location: w, style: b },
                err: null
              };
            }
          }
          return {
            val: {
              type: f === "number" ? ie.number : f === "date" ? ie.date : ie.time,
              value: n,
              location: w,
              style: (s = c?.style) !== null && s !== void 0 ? s : null
            },
            err: null
          };
        }
        case "plural":
        case "selectordinal":
        case "select": {
          var T = this.clonePosition();
          if (this.bumpSpace(), !this.bumpIf(","))
            return this.error(V.EXPECT_SELECT_ARGUMENT_OPTIONS, X(T, Z({}, T)));
          this.bumpSpace();
          var y = this.parseIdentifierIfPossible(), S = 0;
          if (f !== "select" && y.value === "offset") {
            if (!this.bumpIf(":"))
              return this.error(V.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, X(this.clonePosition(), this.clonePosition()));
            this.bumpSpace();
            var _ = this.tryParseDecimalInteger(V.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, V.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE);
            if (_.err)
              return _;
            this.bumpSpace(), y = this.parseIdentifierIfPossible(), S = _.val;
          }
          var B = this.tryParsePluralOrSelectOptions(t, f, r, y);
          if (B.err)
            return B;
          var O = this.tryParseArgumentClose(i);
          if (O.err)
            return O;
          var N = X(i, this.clonePosition());
          return f === "select" ? {
            val: {
              type: ie.select,
              value: n,
              options: Ti(B.val),
              location: N
            },
            err: null
          } : {
            val: {
              type: ie.plural,
              value: n,
              options: Ti(B.val),
              offset: S,
              pluralType: f === "plural" ? "cardinal" : "ordinal",
              location: N
            },
            err: null
          };
        }
        default:
          return this.error(V.INVALID_ARGUMENT_TYPE, X(u, h));
      }
    }, e.prototype.tryParseArgumentClose = function(t) {
      return this.isEOF() || this.char() !== 125 ? this.error(V.EXPECT_ARGUMENT_CLOSING_BRACE, X(t, this.clonePosition())) : (this.bump(), { val: !0, err: null });
    }, e.prototype.parseSimpleArgStyleIfPossible = function() {
      for (var t = 0, r = this.clonePosition(); !this.isEOF(); ) {
        var n = this.char();
        switch (n) {
          case 39: {
            this.bump();
            var i = this.clonePosition();
            if (!this.bumpUntil("'"))
              return this.error(V.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE, X(i, this.clonePosition()));
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
        n = Uo(t);
      } catch {
        return this.error(V.INVALID_NUMBER_SKELETON, r);
      }
      return {
        val: {
          type: zt.number,
          tokens: n,
          location: r,
          parsedOptions: this.shouldParseSkeletons ? Xo(n) : {}
        },
        err: null
      };
    }, e.prototype.tryParsePluralOrSelectOptions = function(t, r, n, i) {
      for (var s, u = !1, f = [], h = /* @__PURE__ */ new Set(), c = i.value, d = i.location; ; ) {
        if (c.length === 0) {
          var _ = this.clonePosition();
          if (r !== "select" && this.bumpIf("=")) {
            var b = this.tryParseDecimalInteger(V.EXPECT_PLURAL_ARGUMENT_SELECTOR, V.INVALID_PLURAL_ARGUMENT_SELECTOR);
            if (b.err)
              return b;
            d = X(_, this.clonePosition()), c = this.message.slice(_.offset, this.offset());
          } else
            break;
        }
        if (h.has(c))
          return this.error(r === "select" ? V.DUPLICATE_SELECT_ARGUMENT_SELECTOR : V.DUPLICATE_PLURAL_ARGUMENT_SELECTOR, d);
        c === "other" && (u = !0), this.bumpSpace();
        var H = this.clonePosition();
        if (!this.bumpIf("{"))
          return this.error(r === "select" ? V.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT : V.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT, X(this.clonePosition(), this.clonePosition()));
        var O = this.parseMessage(t + 1, r, n);
        if (O.err)
          return O;
        var w = this.tryParseArgumentClose(H);
        if (w.err)
          return w;
        f.push([
          c,
          {
            value: O.val,
            location: X(H, this.clonePosition())
          }
        ]), h.add(c), this.bumpSpace(), s = this.parseIdentifierIfPossible(), c = s.value, d = s.location;
      }
      return f.length === 0 ? this.error(r === "select" ? V.EXPECT_SELECT_ARGUMENT_SELECTOR : V.EXPECT_PLURAL_ARGUMENT_SELECTOR, X(this.clonePosition(), this.clonePosition())) : this.requiresOtherClause && !u ? this.error(V.MISSING_OTHER_CLAUSE, X(this.clonePosition(), this.clonePosition())) : { val: f, err: null };
    }, e.prototype.tryParseDecimalInteger = function(t, r) {
      var n = 1, i = this.clonePosition();
      this.bumpIf("+") || this.bumpIf("-") && (n = -1);
      for (var s = !1, u = 0; !this.isEOF(); ) {
        var f = this.char();
        if (f >= 48 && f <= 57)
          s = !0, u = u * 10 + (f - 48), this.bump();
        else
          break;
      }
      var h = X(i, this.clonePosition());
      return s ? (u *= n, nl(u) ? { val: u, err: null } : this.error(r, h)) : this.error(t, h);
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
      var r = ba(this.message, t);
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
      if (wi(this.message, t, this.offset())) {
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
      for (; !this.isEOF() && ya(this.char()); )
        this.bump();
    }, e.prototype.peek = function() {
      if (this.isEOF())
        return null;
      var t = this.char(), r = this.offset(), n = this.message.charCodeAt(r + (t >= 65536 ? 2 : 1));
      return n ?? null;
    }, e;
  })()
);
function On(e) {
  return e >= 97 && e <= 122 || e >= 65 && e <= 90;
}
function ll(e) {
  return On(e) || e === 47;
}
function ul(e) {
  return e === 45 || e === 46 || e >= 48 && e <= 57 || e === 95 || e >= 97 && e <= 122 || e >= 65 && e <= 90 || e == 183 || e >= 192 && e <= 214 || e >= 216 && e <= 246 || e >= 248 && e <= 893 || e >= 895 && e <= 8191 || e >= 8204 && e <= 8205 || e >= 8255 && e <= 8256 || e >= 8304 && e <= 8591 || e >= 11264 && e <= 12271 || e >= 12289 && e <= 55295 || e >= 63744 && e <= 64975 || e >= 65008 && e <= 65533 || e >= 65536 && e <= 983039;
}
function ya(e) {
  return e >= 9 && e <= 13 || e === 32 || e === 133 || e >= 8206 && e <= 8207 || e === 8232 || e === 8233;
}
function fl(e) {
  return e >= 33 && e <= 35 || e === 36 || e >= 37 && e <= 39 || e === 40 || e === 41 || e === 42 || e === 43 || e === 44 || e === 45 || e >= 46 && e <= 47 || e >= 58 && e <= 59 || e >= 60 && e <= 62 || e >= 63 && e <= 64 || e === 91 || e === 92 || e === 93 || e === 94 || e === 96 || e === 123 || e === 124 || e === 125 || e === 126 || e === 161 || e >= 162 && e <= 165 || e === 166 || e === 167 || e === 169 || e === 171 || e === 172 || e === 174 || e === 176 || e === 177 || e === 182 || e === 187 || e === 191 || e === 215 || e === 247 || e >= 8208 && e <= 8213 || e >= 8214 && e <= 8215 || e === 8216 || e === 8217 || e === 8218 || e >= 8219 && e <= 8220 || e === 8221 || e === 8222 || e === 8223 || e >= 8224 && e <= 8231 || e >= 8240 && e <= 8248 || e === 8249 || e === 8250 || e >= 8251 && e <= 8254 || e >= 8257 && e <= 8259 || e === 8260 || e === 8261 || e === 8262 || e >= 8263 && e <= 8273 || e === 8274 || e === 8275 || e >= 8277 && e <= 8286 || e >= 8592 && e <= 8596 || e >= 8597 && e <= 8601 || e >= 8602 && e <= 8603 || e >= 8604 && e <= 8607 || e === 8608 || e >= 8609 && e <= 8610 || e === 8611 || e >= 8612 && e <= 8613 || e === 8614 || e >= 8615 && e <= 8621 || e === 8622 || e >= 8623 && e <= 8653 || e >= 8654 && e <= 8655 || e >= 8656 && e <= 8657 || e === 8658 || e === 8659 || e === 8660 || e >= 8661 && e <= 8691 || e >= 8692 && e <= 8959 || e >= 8960 && e <= 8967 || e === 8968 || e === 8969 || e === 8970 || e === 8971 || e >= 8972 && e <= 8991 || e >= 8992 && e <= 8993 || e >= 8994 && e <= 9e3 || e === 9001 || e === 9002 || e >= 9003 && e <= 9083 || e === 9084 || e >= 9085 && e <= 9114 || e >= 9115 && e <= 9139 || e >= 9140 && e <= 9179 || e >= 9180 && e <= 9185 || e >= 9186 && e <= 9254 || e >= 9255 && e <= 9279 || e >= 9280 && e <= 9290 || e >= 9291 && e <= 9311 || e >= 9472 && e <= 9654 || e === 9655 || e >= 9656 && e <= 9664 || e === 9665 || e >= 9666 && e <= 9719 || e >= 9720 && e <= 9727 || e >= 9728 && e <= 9838 || e === 9839 || e >= 9840 && e <= 10087 || e === 10088 || e === 10089 || e === 10090 || e === 10091 || e === 10092 || e === 10093 || e === 10094 || e === 10095 || e === 10096 || e === 10097 || e === 10098 || e === 10099 || e === 10100 || e === 10101 || e >= 10132 && e <= 10175 || e >= 10176 && e <= 10180 || e === 10181 || e === 10182 || e >= 10183 && e <= 10213 || e === 10214 || e === 10215 || e === 10216 || e === 10217 || e === 10218 || e === 10219 || e === 10220 || e === 10221 || e === 10222 || e === 10223 || e >= 10224 && e <= 10239 || e >= 10240 && e <= 10495 || e >= 10496 && e <= 10626 || e === 10627 || e === 10628 || e === 10629 || e === 10630 || e === 10631 || e === 10632 || e === 10633 || e === 10634 || e === 10635 || e === 10636 || e === 10637 || e === 10638 || e === 10639 || e === 10640 || e === 10641 || e === 10642 || e === 10643 || e === 10644 || e === 10645 || e === 10646 || e === 10647 || e === 10648 || e >= 10649 && e <= 10711 || e === 10712 || e === 10713 || e === 10714 || e === 10715 || e >= 10716 && e <= 10747 || e === 10748 || e === 10749 || e >= 10750 && e <= 11007 || e >= 11008 && e <= 11055 || e >= 11056 && e <= 11076 || e >= 11077 && e <= 11078 || e >= 11079 && e <= 11084 || e >= 11085 && e <= 11123 || e >= 11124 && e <= 11125 || e >= 11126 && e <= 11157 || e === 11158 || e >= 11159 && e <= 11263 || e >= 11776 && e <= 11777 || e === 11778 || e === 11779 || e === 11780 || e === 11781 || e >= 11782 && e <= 11784 || e === 11785 || e === 11786 || e === 11787 || e === 11788 || e === 11789 || e >= 11790 && e <= 11798 || e === 11799 || e >= 11800 && e <= 11801 || e === 11802 || e === 11803 || e === 11804 || e === 11805 || e >= 11806 && e <= 11807 || e === 11808 || e === 11809 || e === 11810 || e === 11811 || e === 11812 || e === 11813 || e === 11814 || e === 11815 || e === 11816 || e === 11817 || e >= 11818 && e <= 11822 || e === 11823 || e >= 11824 && e <= 11833 || e >= 11834 && e <= 11835 || e >= 11836 && e <= 11839 || e === 11840 || e === 11841 || e === 11842 || e >= 11843 && e <= 11855 || e >= 11856 && e <= 11857 || e === 11858 || e >= 11859 && e <= 11903 || e >= 12289 && e <= 12291 || e === 12296 || e === 12297 || e === 12298 || e === 12299 || e === 12300 || e === 12301 || e === 12302 || e === 12303 || e === 12304 || e === 12305 || e >= 12306 && e <= 12307 || e === 12308 || e === 12309 || e === 12310 || e === 12311 || e === 12312 || e === 12313 || e === 12314 || e === 12315 || e === 12316 || e === 12317 || e >= 12318 && e <= 12319 || e === 12320 || e === 12336 || e === 64830 || e === 64831 || e >= 65093 && e <= 65094;
}
function Nn(e) {
  e.forEach(function(t) {
    if (delete t.location, fa(t) || ca(t))
      for (var r in t.options)
        delete t.options[r].location, Nn(t.options[r].value);
    else oa(t) && da(t.style) || (la(t) || ua(t)) && An(t.style) ? delete t.style.location : ha(t) && Nn(t.children);
  });
}
function cl(e, t) {
  t === void 0 && (t = {}), t = Z({ shouldParseSkeletons: !0, requiresOtherClause: !0 }, t);
  var r = new ol(e, t).parse();
  if (r.err) {
    var n = SyntaxError(V[r.err.kind]);
    throw n.location = r.err.location, n.originalMessage = r.err.message, n;
  }
  return t?.captureLocation || Nn(r.val), r.val;
}
var Xt;
(function(e) {
  e.MISSING_VALUE = "MISSING_VALUE", e.INVALID_VALUE = "INVALID_VALUE", e.MISSING_INTL_API = "MISSING_INTL_API";
})(Xt || (Xt = {}));
var Xr = (
  /** @class */
  (function(e) {
    zr(t, e);
    function t(r, n, i) {
      var s = e.call(this, r) || this;
      return s.code = n, s.originalMessage = i, s;
    }
    return t.prototype.toString = function() {
      return "[formatjs Error: ".concat(this.code, "] ").concat(this.message);
    }, t;
  })(Error)
), Ai = (
  /** @class */
  (function(e) {
    zr(t, e);
    function t(r, n, i, s) {
      return e.call(this, 'Invalid values for "'.concat(r, '": "').concat(n, '". Options are "').concat(Object.keys(i).join('", "'), '"'), Xt.INVALID_VALUE, s) || this;
    }
    return t;
  })(Xr)
), hl = (
  /** @class */
  (function(e) {
    zr(t, e);
    function t(r, n, i) {
      return e.call(this, 'Value for "'.concat(r, '" must be of type ').concat(n), Xt.INVALID_VALUE, i) || this;
    }
    return t;
  })(Xr)
), dl = (
  /** @class */
  (function(e) {
    zr(t, e);
    function t(r, n) {
      return e.call(this, 'The intl string context variable "'.concat(r, '" was not provided to the string "').concat(n, '"'), Xt.MISSING_VALUE, n) || this;
    }
    return t;
  })(Xr)
), Ie;
(function(e) {
  e[e.literal = 0] = "literal", e[e.object = 1] = "object";
})(Ie || (Ie = {}));
function pl(e) {
  return e.length < 2 ? e : e.reduce(function(t, r) {
    var n = t[t.length - 1];
    return !n || n.type !== Ie.literal || r.type !== Ie.literal ? t.push(r) : n.value += r.value, t;
  }, []);
}
function vl(e) {
  return typeof e == "function";
}
function Nr(e, t, r, n, i, s, u) {
  if (e.length === 1 && _i(e[0]))
    return [
      {
        type: Ie.literal,
        value: e[0].value
      }
    ];
  for (var f = [], h = 0, c = e; h < c.length; h++) {
    var d = c[h];
    if (_i(d)) {
      f.push({
        type: Ie.literal,
        value: d.value
      });
      continue;
    }
    if (ko(d)) {
      typeof s == "number" && f.push({
        type: Ie.literal,
        value: r.getNumberFormat(t).format(s)
      });
      continue;
    }
    var _ = d.value;
    if (!(i && _ in i))
      throw new dl(_, u);
    var b = i[_];
    if (Ro(d)) {
      (!b || typeof b == "string" || typeof b == "number") && (b = typeof b == "string" || typeof b == "number" ? String(b) : ""), f.push({
        type: typeof b == "string" ? Ie.literal : Ie.object,
        value: b
      });
      continue;
    }
    if (la(d)) {
      var H = typeof d.style == "string" ? n.date[d.style] : An(d.style) ? d.style.parsedOptions : void 0;
      f.push({
        type: Ie.literal,
        value: r.getDateTimeFormat(t, H).format(b)
      });
      continue;
    }
    if (ua(d)) {
      var H = typeof d.style == "string" ? n.time[d.style] : An(d.style) ? d.style.parsedOptions : n.time.medium;
      f.push({
        type: Ie.literal,
        value: r.getDateTimeFormat(t, H).format(b)
      });
      continue;
    }
    if (oa(d)) {
      var H = typeof d.style == "string" ? n.number[d.style] : da(d.style) ? d.style.parsedOptions : void 0;
      H && H.scale && (b = b * (H.scale || 1)), f.push({
        type: Ie.literal,
        value: r.getNumberFormat(t, H).format(b)
      });
      continue;
    }
    if (ha(d)) {
      var O = d.children, w = d.value, m = i[w];
      if (!vl(m))
        throw new hl(w, "function", u);
      var v = Nr(O, t, r, n, i, s), E = m(v.map(function(S) {
        return S.value;
      }));
      Array.isArray(E) || (E = [E]), f.push.apply(f, E.map(function(S) {
        return {
          type: typeof S == "string" ? Ie.literal : Ie.object,
          value: S
        };
      }));
    }
    if (fa(d)) {
      var T = d.options[b] || d.options.other;
      if (!T)
        throw new Ai(d.value, b, Object.keys(d.options), u);
      f.push.apply(f, Nr(T.value, t, r, n, i));
      continue;
    }
    if (ca(d)) {
      var T = d.options["=".concat(b)];
      if (!T) {
        if (!Intl.PluralRules)
          throw new Xr(`Intl.PluralRules is not available in this environment.
Try polyfilling it using "@formatjs/intl-pluralrules"
`, Xt.MISSING_INTL_API, u);
        var y = r.getPluralRules(t, { type: d.pluralType }).select(b - (d.offset || 0));
        T = d.options[y] || d.options.other;
      }
      if (!T)
        throw new Ai(d.value, b, Object.keys(d.options), u);
      f.push.apply(f, Nr(T.value, t, r, n, i, b - (d.offset || 0)));
      continue;
    }
  }
  return pl(f);
}
function ml(e, t) {
  return t ? Z(Z(Z({}, e || {}), t || {}), Object.keys(e).reduce(function(r, n) {
    return r[n] = Z(Z({}, e[n]), t[n] || {}), r;
  }, {})) : e;
}
function gl(e, t) {
  return t ? Object.keys(e).reduce(function(r, n) {
    return r[n] = ml(e[n], t[n]), r;
  }, Z({}, e)) : e;
}
function dn(e) {
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
function bl(e) {
  return e === void 0 && (e = {
    number: {},
    dateTime: {},
    pluralRules: {}
  }), {
    getNumberFormat: fn(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.NumberFormat).bind.apply(t, un([void 0], r, !1)))();
    }, {
      cache: dn(e.number),
      strategy: cn.variadic
    }),
    getDateTimeFormat: fn(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.DateTimeFormat).bind.apply(t, un([void 0], r, !1)))();
    }, {
      cache: dn(e.dateTime),
      strategy: cn.variadic
    }),
    getPluralRules: fn(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.PluralRules).bind.apply(t, un([void 0], r, !1)))();
    }, {
      cache: dn(e.pluralRules),
      strategy: cn.variadic
    })
  };
}
var _l = (
  /** @class */
  (function() {
    function e(t, r, n, i) {
      r === void 0 && (r = e.defaultLocale);
      var s = this;
      if (this.formatterCache = {
        number: {},
        dateTime: {},
        pluralRules: {}
      }, this.format = function(h) {
        var c = s.formatToParts(h);
        if (c.length === 1)
          return c[0].value;
        var d = c.reduce(function(_, b) {
          return !_.length || b.type !== Ie.literal || typeof _[_.length - 1] != "string" ? _.push(b.value) : _[_.length - 1] += b.value, _;
        }, []);
        return d.length <= 1 ? d[0] || "" : d;
      }, this.formatToParts = function(h) {
        return Nr(s.ast, s.locales, s.formatters, s.formats, h, void 0, s.message);
      }, this.resolvedOptions = function() {
        var h;
        return {
          locale: ((h = s.resolvedLocale) === null || h === void 0 ? void 0 : h.toString()) || Intl.NumberFormat.supportedLocalesOf(s.locales)[0]
        };
      }, this.getAst = function() {
        return s.ast;
      }, this.locales = r, this.resolvedLocale = e.resolveLocale(r), typeof t == "string") {
        if (this.message = t, !e.__parse)
          throw new TypeError("IntlMessageFormat.__parse must be set to process `message` of type `string`");
        var u = i || {};
        u.formatters;
        var f = Mo(u, ["formatters"]);
        this.ast = e.__parse(t, Z(Z({}, f), { locale: this.resolvedLocale }));
      } else
        this.ast = t;
      if (!Array.isArray(this.ast))
        throw new TypeError("A message must be provided as a String or AST.");
      this.formats = gl(e.formats, n), this.formatters = i && i.formatters || bl(this.formatterCache);
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
    }, e.__parse = cl, e.formats = {
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
function yl(e, t) {
  if (t == null)
    return;
  if (t in e)
    return e[t];
  const r = t.split(".");
  let n = e;
  for (let i = 0; i < r.length; i++)
    if (typeof n == "object") {
      if (i > 0) {
        const s = r.slice(i, r.length).join(".");
        if (s in n) {
          n = n[s];
          break;
        }
      }
      n = n[r[i]];
    } else
      n = void 0;
  return n;
}
const yt = {}, xl = (e, t, r) => r && (t in yt || (yt[t] = {}), e in yt[t] || (yt[t][e] = r), r), xa = (e, t) => {
  if (t == null)
    return;
  if (t in yt && e in yt[t])
    return yt[t][e];
  const r = Wr(t);
  for (let n = 0; n < r.length; n++) {
    const i = r[n], s = wl(i, e);
    if (s)
      return xl(e, t, s);
  }
};
let Vn;
const cr = fr({});
function El(e) {
  return Vn[e] || null;
}
function Ea(e) {
  return e in Vn;
}
function wl(e, t) {
  if (!Ea(e))
    return null;
  const r = El(e);
  return yl(r, t);
}
function Tl(e) {
  if (e == null)
    return;
  const t = Wr(e);
  for (let r = 0; r < t.length; r++) {
    const n = t[r];
    if (Ea(n))
      return n;
  }
}
function Sl(e, ...t) {
  delete yt[e], cr.update((r) => (r[e] = Ho.all([r[e] || {}, ...t]), r));
}
qt(
  [cr],
  ([e]) => Object.keys(e)
);
cr.subscribe((e) => Vn = e);
const Ir = {};
function Al(e, t) {
  Ir[e].delete(t), Ir[e].size === 0 && delete Ir[e];
}
function wa(e) {
  return Ir[e];
}
function Hl(e) {
  return Wr(e).map((t) => {
    const r = wa(t);
    return [t, r ? [...r] : []];
  }).filter(([, t]) => t.length > 0);
}
function In(e) {
  return e == null ? !1 : Wr(e).some(
    (t) => {
      var r;
      return (r = wa(t)) == null ? void 0 : r.size;
    }
  );
}
function Ml(e, t) {
  return Promise.all(
    t.map((n) => (Al(e, n), n().then((i) => i.default || i)))
  ).then((n) => Sl(e, ...n));
}
const nr = {};
function Ta(e) {
  if (!In(e))
    return e in nr ? nr[e] : Promise.resolve();
  const t = Hl(e);
  return nr[e] = Promise.all(
    t.map(
      ([r, n]) => Ml(r, n)
    )
  ).then(() => {
    if (In(e))
      return Ta(e);
    delete nr[e];
  }), nr[e];
}
const Pl = {
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
}, Ol = {
  fallbackLocale: null,
  loadingDelay: 200,
  formats: Pl,
  warnOnMissingMessages: !0,
  handleMissingMessage: void 0,
  ignoreTag: !0
}, Nl = Ol;
function Wt() {
  return Nl;
}
const pn = fr(!1);
var Il = Object.defineProperty, Ll = Object.defineProperties, Bl = Object.getOwnPropertyDescriptors, Hi = Object.getOwnPropertySymbols, Cl = Object.prototype.hasOwnProperty, Rl = Object.prototype.propertyIsEnumerable, Mi = (e, t, r) => t in e ? Il(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, kl = (e, t) => {
  for (var r in t || (t = {}))
    Cl.call(t, r) && Mi(e, r, t[r]);
  if (Hi)
    for (var r of Hi(t))
      Rl.call(t, r) && Mi(e, r, t[r]);
  return e;
}, Dl = (e, t) => Ll(e, Bl(t));
let Ln;
const Rr = fr(null);
function Pi(e) {
  return e.split("-").map((t, r, n) => n.slice(0, r + 1).join("-")).reverse();
}
function Wr(e, t = Wt().fallbackLocale) {
  const r = Pi(e);
  return t ? [.../* @__PURE__ */ new Set([...r, ...Pi(t)])] : r;
}
function Nt() {
  return Ln ?? void 0;
}
Rr.subscribe((e) => {
  Ln = e ?? void 0, typeof window < "u" && e != null && document.documentElement.setAttribute("lang", e);
});
const Gl = (e) => {
  if (e && Tl(e) && In(e)) {
    const { loadingDelay: t } = Wt();
    let r;
    return typeof window < "u" && Nt() != null && t ? r = window.setTimeout(
      () => pn.set(!0),
      t
    ) : pn.set(!0), Ta(e).then(() => {
      Rr.set(e);
    }).finally(() => {
      clearTimeout(r), pn.set(!1);
    });
  }
  return Rr.set(e);
}, Zt = Dl(kl({}, Rr), {
  set: Gl
}), qr = (e) => {
  const t = /* @__PURE__ */ Object.create(null);
  return (n) => {
    const i = JSON.stringify(n);
    return i in t ? t[i] : t[i] = e(n);
  };
};
var Fl = Object.defineProperty, kr = Object.getOwnPropertySymbols, Sa = Object.prototype.hasOwnProperty, Aa = Object.prototype.propertyIsEnumerable, Oi = (e, t, r) => t in e ? Fl(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, zn = (e, t) => {
  for (var r in t || (t = {}))
    Sa.call(t, r) && Oi(e, r, t[r]);
  if (kr)
    for (var r of kr(t))
      Aa.call(t, r) && Oi(e, r, t[r]);
  return e;
}, Yt = (e, t) => {
  var r = {};
  for (var n in e)
    Sa.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && kr)
    for (var n of kr(e))
      t.indexOf(n) < 0 && Aa.call(e, n) && (r[n] = e[n]);
  return r;
};
const lr = (e, t) => {
  const { formats: r } = Wt();
  if (e in r && t in r[e])
    return r[e][t];
  throw new Error(`[svelte-i18n] Unknown "${t}" ${e} format.`);
}, Ul = qr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = Yt(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format numbers');
    return n && (i = lr("number", n)), new Intl.NumberFormat(r, i);
  }
), jl = qr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = Yt(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format dates');
    return n ? i = lr("date", n) : Object.keys(i).length === 0 && (i = lr("date", "short")), new Intl.DateTimeFormat(r, i);
  }
), Vl = qr(
  (e) => {
    var t = e, { locale: r, format: n } = t, i = Yt(t, ["locale", "format"]);
    if (r == null)
      throw new Error(
        '[svelte-i18n] A "locale" must be set to format time values'
      );
    return n ? i = lr("time", n) : Object.keys(i).length === 0 && (i = lr("time", "short")), new Intl.DateTimeFormat(r, i);
  }
), zl = (e = {}) => {
  var t = e, {
    locale: r = Nt()
  } = t, n = Yt(t, [
    "locale"
  ]);
  return Ul(zn({ locale: r }, n));
}, Xl = (e = {}) => {
  var t = e, {
    locale: r = Nt()
  } = t, n = Yt(t, [
    "locale"
  ]);
  return jl(zn({ locale: r }, n));
}, Wl = (e = {}) => {
  var t = e, {
    locale: r = Nt()
  } = t, n = Yt(t, [
    "locale"
  ]);
  return Vl(zn({ locale: r }, n));
}, ql = qr(
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  (e, t = Nt()) => new _l(e, t, Wt().formats, {
    ignoreTag: Wt().ignoreTag
  })
), Zl = (e, t = {}) => {
  var r, n, i, s;
  let u = t;
  typeof e == "object" && (u = e, e = u.id);
  const {
    values: f,
    locale: h = Nt(),
    default: c
  } = u;
  if (h == null)
    throw new Error(
      "[svelte-i18n] Cannot format a message without first setting the initial locale."
    );
  let d = xa(e, h);
  if (!d)
    d = (s = (i = (n = (r = Wt()).handleMissingMessage) == null ? void 0 : n.call(r, { locale: h, id: e, defaultValue: c })) != null ? i : c) != null ? s : e;
  else if (typeof d != "string")
    return console.warn(
      `[svelte-i18n] Message with id "${e}" must be of type "string", found: "${typeof d}". Gettin its value through the "$format" method is deprecated; use the "json" method instead.`
    ), d;
  if (!f)
    return d;
  let _ = d;
  try {
    _ = ql(d, h).format(f);
  } catch (b) {
    b instanceof Error && console.warn(
      `[svelte-i18n] Message "${e}" has syntax error:`,
      b.message
    );
  }
  return _;
}, Yl = (e, t) => Wl(t).format(e), Jl = (e, t) => Xl(t).format(e), Ql = (e, t) => zl(t).format(e), Kl = (e, t = Nt()) => xa(e, t);
qt([Zt, cr], () => Zl);
qt([Zt], () => Yl);
qt([Zt], () => Jl);
qt([Zt], () => Ql);
qt([Zt, cr], () => Kl);
const $l = "__i18n__", eu = [
  "label",
  "info",
  "placeholder",
  "description",
  "title",
  "value"
], tu = [
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
function ru(e) {
  return typeof e == "string" && e.includes($l);
}
class nu {
  load_component;
  #t = Y(or({}));
  get shared() {
    return o(this.#t);
  }
  set shared(t) {
    x(this.#t, t, !0);
  }
  #r = Y(or({}));
  get props() {
    return o(this.#r);
  }
  set props(t) {
    x(this.#r, t, !0);
  }
  #e = Y((t) => t);
  get i18n() {
    return o(this.#e);
  }
  set i18n(t) {
    x(this.#e, t, !0);
  }
  translatable_props = {};
  dispatcher;
  last_update = null;
  shared_props = tu;
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
    for (const n of eu)
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
    ), We(() => {
      for (const n in t.shared_props)
        this._is_i18n_managed(`shared.${n}`, t.shared_props[n]) || (this.shared[n] = t.shared_props[n]);
      for (const n in t.props)
        this._is_i18n_managed(`props.${n}`, t.props[n]) || (this.props[n] = t.props[n]);
      this.register_component(
        t.shared_props.id,
        // @ts-ignore
        this.set_data.bind(this),
        this.get_data.bind(this)
      ), xe(() => {
        this.shared.id = t.shared_props.id;
      });
    }), Object.keys(this.translatable_props).length > 0 && Zt.subscribe(() => {
      for (const [n, i] of Object.entries(this.translatable_props)) {
        const [s, u] = n.split("."), f = this.i18n(i);
        s === "shared" ? this.shared[u] = f : this.props[u] = f;
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
    return Fs(this.props);
  }
  update(t) {
    this.set_data(t);
  }
  set_data(t) {
    for (const r in t) {
      const n = t[r], i = ru(n) ? this._translate_and_store(this.shared_props.includes(r) ? "shared" : "props", r, n) : n;
      if (this.shared_props.includes(r)) {
        const s = r;
        this.shared[s] = i;
        continue;
      }
      this.props[r] = i;
    }
  }
  watch_for_change() {
    We(() => {
      this.mounted || (this.old_value = this.props.value, this.mounted = !0), this.old_value != this.props.value && (this.old_value = this.props.value, this.dispatch("change"));
    });
  }
}
Cs();
var iu = /* @__PURE__ */ Qi('<svg class="resize-handle svelte-1stq1b1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><line x1="1" y1="9" x2="9" y2="1" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line><line x1="5" y1="9" x2="9" y2="5" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line></svg>'), Ni = /* @__PURE__ */ Ee("<!> <!>", 1), au = /* @__PURE__ */ Ee('<div class="placeholder svelte-1stq1b1"></div>');
function su(e, t) {
  jr(t, !1);
  let r = C(t, "height", 8, void 0), n = C(t, "min_height", 8, void 0), i = C(t, "max_height", 8, void 0), s = C(t, "width", 8, void 0), u = C(t, "elem_id", 8, ""), f = C(t, "elem_classes", 24, () => []), h = C(t, "variant", 8, "solid"), c = C(t, "border_mode", 8, "base"), d = C(t, "padding", 8, !0), _ = C(t, "type", 8, "normal"), b = C(t, "test_id", 8, void 0), H = C(t, "explicit_call", 8, !1), O = C(t, "container", 8, !0), w = C(t, "visible", 8, !0), m = C(t, "allow_overflow", 8, !0), v = C(t, "overflow_behavior", 8, "auto"), E = C(t, "scale", 8, null), T = C(t, "min_width", 8, 0), y = C(t, "flex", 12, !1), S = C(t, "resizable", 8, !1), B = C(t, "rtl", 8, !1), N = C(t, "fullscreen", 12, !1), k = C(t, "label", 8, void 0), U = Mt(N()), R = Mt(), $ = _() === "fieldset" ? "fieldset" : "div", D = Mt(0), P = Mt(0), I = Mt(null);
  function le(he) {
    N() && he.key === "Escape" && N(!1);
  }
  const ue = (he) => {
    if (he !== void 0) {
      if (typeof he == "number")
        return he + "px";
      if (typeof he == "string")
        return he;
    }
  }, be = (he) => {
    let Ge = he.clientY;
    const ne = (fe) => {
      const _e = fe.clientY - Ge;
      Ge = fe.clientY, ks(R, o(R).style.height = `${o(R).offsetHeight + _e}px`);
    }, Ye = () => {
      window.removeEventListener("mousemove", ne), window.removeEventListener("mouseup", Ye);
    };
    window.addEventListener("mousemove", ne), window.addEventListener("mouseup", Ye);
  };
  ui(
    () => (ze(N()), o(U), o(R)),
    () => {
      N() !== o(U) && (x(U, N()), N() ? (x(I, o(R).getBoundingClientRect()), x(D, o(R).offsetHeight), x(P, o(R).offsetWidth), window.addEventListener("keydown", le)) : (x(I, null), window.removeEventListener("keydown", le)));
    }
  ), ui(() => ze(w()), () => {
    w() || y(!1);
  }), Rs(), bo();
  var Le = Ut(), Ze = Ne(Le);
  {
    var dt = (he) => {
      var Ge = Ni(), ne = Ne(Ge);
      ao(ne, () => $, !1, (_e, tt) => {
        sr(_e, (ye) => x(R, ye), () => o(R)), go(
          _e,
          (ye, Fe) => ({
            "data-testid": b(),
            id: u(),
            class: `block ${ye ?? ""}`,
            dir: B() ? "rtl" : "ltr",
            "aria-label": k(),
            style: "",
            [ar]: {
              hidden: w() === "hidden",
              padded: d(),
              flex: y(),
              border_focus: c() === "focus",
              border_contrast: c() === "contrast",
              "hide-container": !H() && !O(),
              fullscreen: N(),
              animating: N() && o(I) !== null,
              "auto-margin": E() === null
            },
            [Ft]: Fe
          }),
          [
            () => (ze(f()), xe(() => f()?.join(" ") || "")),
            () => ({
              height: (ze(N()), ze(r()), xe(() => N() ? void 0 : ue(r()))),
              "min-height": (ze(N()), ze(n()), xe(() => N() ? void 0 : ue(n()))),
              "max-height": (ze(N()), ze(i()), xe(() => N() ? void 0 : ue(i()))),
              "--start-top": (o(I), xe(() => o(I) ? `${o(I).top}px` : "0px")),
              "--start-left": (o(I), xe(() => o(I) ? `${o(I).left}px` : "0px")),
              "--start-width": (o(I), xe(() => o(I) ? `${o(I).width}px` : "0px")),
              "--start-height": (o(I), xe(() => o(I) ? `${o(I).height}px` : "0px")),
              width: (ze(N()), ze(s()), xe(() => N() ? void 0 : typeof s() == "number" ? `calc(min(${s()}px, 100%))` : ue(s()))),
              "border-style": h(),
              overflow: m() ? v() : "hidden",
              "flex-grow": E(),
              "min-width": `calc(min(${T()}px, 100%))`
            })
          ],
          void 0,
          void 0,
          "svelte-1stq1b1"
        );
        var Ce = Ni(), ut = Ne(Ce);
        Tn(ut, t, "default", {});
        var Je = z(ut, 2);
        {
          var Re = (ye) => {
            var Fe = iu();
            Oe("mousedown", Fe, be), j(ye, Fe);
          };
          se(Je, (ye) => {
            S() && ye(Re);
          });
        }
        j(tt, Ce);
      });
      var Ye = z(ne, 2);
      {
        var fe = (_e) => {
          var tt = au();
          let Ce;
          oe(() => Ce = qe(tt, "", Ce, {
            height: o(D) + "px",
            width: o(P) + "px"
          })), j(_e, tt);
        };
        se(Ye, (_e) => {
          N() && _e(fe);
        });
      }
      j(he, Ge);
    };
    se(Ze, (he) => {
      (w() === !0 || w() === "hidden") && he(dt);
    });
  }
  j(e, Le), Ur();
}
var ou = /* @__PURE__ */ Ee('<span class="svelte-vvirtv"> </span>'), lu = /* @__PURE__ */ Ee("<button><!> <div><!> <!></div></button>");
function Ii(e, t) {
  let r = C(t, "label", 3, ""), n = C(t, "show_label", 3, !1), i = C(t, "pending", 3, !1), s = C(t, "size", 3, "small"), u = C(t, "padded", 3, !0), f = C(t, "highlight", 3, !1), h = C(t, "disabled", 3, !1), c = C(t, "hasPopup", 3, !1), d = C(t, "color", 3, "var(--block-label-text-color)"), _ = C(t, "transparent", 3, !1), b = C(t, "background", 3, "var(--block-background-fill)"), H = C(t, "border", 3, "transparent"), O = Xe(() => f() ? "var(--color-accent)" : d());
  var w = lu();
  let m, v;
  var E = te(w);
  {
    var T = (U) => {
      var R = ou(), $ = te(R);
      oe(() => ge($, r())), j(U, R);
    };
    se(E, (U) => {
      n() && U(T);
    });
  }
  var y = z(E, 2);
  let S;
  var B = te(y);
  ro(B, () => t.Icon, (U, R) => {
    R(U, {});
  });
  var N = z(B, 2);
  {
    var k = (U) => {
      var R = Ut(), $ = Ne(R);
      Ys($, () => t.children), j(U, R);
    };
    se(N, (U) => {
      t.children && U(k);
    });
  }
  oe(() => {
    m = ht(w, 1, "icon-button svelte-vvirtv", null, m, {
      pending: i(),
      padded: u(),
      highlight: f(),
      transparent: _()
    }), w.disabled = h(), wt(w, "aria-label", r()), wt(w, "aria-haspopup", c()), wt(w, "title", r()), v = qe(w, "", v, {
      "--border-color": H(),
      color: !h() && o(O) ? o(O) : "var(--block-label-text-color)",
      "--bg-color": h() ? "auto" : b()
    }), S = ht(y, 1, "svelte-vvirtv", null, S, {
      "x-small": s() === "x-small",
      small: s() === "small",
      large: s() === "large",
      medium: s() === "medium"
    });
  }), qi("click", w, function(...U) {
    t.onclick?.apply(this, U);
  }), j(e, w);
}
Fr(["click"]);
var uu = /* @__PURE__ */ Qi('<svg width="100%" height="100%" viewBox="0 0 24 24" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:space="preserve" stroke="currentColor" style="fill-rule:evenodd;clip-rule:evenodd;stroke-linecap:round;stroke-linejoin:round;"><g transform="matrix(1.14096,-0.140958,-0.140958,1.14096,-0.0559523,0.0559523)"><path d="M18,6L6.087,17.913" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></g><path d="M4.364,4.364L19.636,19.636" style="fill:none;fill-rule:nonzero;stroke-width:2px;"></path></svg>');
function Li(e) {
  var t = uu();
  j(e, t);
}
Fr(["click"]);
function vn(e) {
  let t = ["", "k", "M", "G", "T", "P", "E", "Z"], r = 0;
  for (; e > 1e3 && r < t.length - 1; )
    e /= 1e3, r++;
  let n = t[r];
  return (Number.isInteger(e) ? e : e.toFixed(1)) + n;
}
function Bi(e) {
  return Object.prototype.toString.call(e) === "[object Date]";
}
function Bn(e, t, r, n) {
  if (typeof r == "number" || Bi(r)) {
    const i = n - r, s = (r - t) / (e.dt || 1 / 60), u = e.opts.stiffness * i, f = e.opts.damping * s, h = (u - f) * e.inv_mass, c = (s + h) * e.dt;
    return Math.abs(c) < e.opts.precision && Math.abs(i) < e.opts.precision ? n : (e.settled = !1, Bi(r) ? new Date(r.getTime() + c) : r + c);
  } else {
    if (Array.isArray(r))
      return r.map(
        (i, s) => (
          // @ts-ignore
          Bn(e, t[s], r[s], n[s])
        )
      );
    if (typeof r == "object") {
      const i = {};
      for (const s in r)
        i[s] = Bn(e, t[s], r[s], n[s]);
      return i;
    } else
      throw new Error(`Cannot spring ${typeof r} values`);
  }
}
function Ci(e, t = {}) {
  const r = fr(e), { stiffness: n = 0.15, damping: i = 0.8, precision: s = 0.01 } = t;
  let u, f, h, c = (
    /** @type {T} */
    e
  ), d = (
    /** @type {T | undefined} */
    e
  ), _ = 1, b = 0, H = !1;
  function O(m, v = {}) {
    d = m;
    const E = h = {};
    return e == null || v.hard || w.stiffness >= 1 && w.damping >= 1 ? (H = !0, u = et.now(), c = m, r.set(e = d), Promise.resolve()) : (v.soft && (b = 1 / ((v.soft === !0 ? 0.5 : +v.soft) * 60), _ = 0), f || (u = et.now(), H = !1, f = io((T) => {
      if (H)
        return H = !1, f = null, !1;
      _ = Math.min(_ + b, 1);
      const y = Math.min(T - u, 1e3 / 30), S = {
        inv_mass: _,
        opts: w,
        settled: !0,
        dt: y * 60 / 1e3
      }, B = Bn(S, c, e, d);
      return u = T, c = /** @type {T} */
      e, r.set(e = /** @type {T} */
      B), S.settled && (f = null), !S.settled;
    })), new Promise((T) => {
      f.promise.then(() => {
        E === h && T();
      });
    }));
  }
  const w = {
    set: O,
    update: (m, v) => O(m(
      /** @type {T} */
      d,
      /** @type {T} */
      e
    ), v),
    subscribe: r.subscribe,
    stiffness: n,
    damping: i,
    precision: s
  };
  return w;
}
var fu = /* @__PURE__ */ Ee('<div><svg viewBox="-1200 -1200 3000 3000" fill="none" xmlns="http://www.w3.org/2000/svg" class="svelte-m6d381"><g><path d="M255.926 0.754768L509.702 139.936V221.027L255.926 81.8465V0.754768Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M509.69 139.936L254.981 279.641V361.255L509.69 221.55V139.936Z" fill="#FF7C00" class="svelte-m6d381"></path><path d="M0.250138 139.937L254.981 279.641V361.255L0.250138 221.55V139.937Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M255.923 0.232622L0.236328 139.936V221.55L255.923 81.8469V0.232622Z" fill="#FF7C00" class="svelte-m6d381"></path></g><g><path d="M255.926 141.5L509.702 280.681V361.773L255.926 222.592V141.5Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M509.69 280.679L254.981 420.384V501.998L509.69 362.293V280.679Z" fill="#FF7C00" class="svelte-m6d381"></path><path d="M0.250138 280.681L254.981 420.386V502L0.250138 362.295V280.681Z" fill="#FF7C00" fill-opacity="0.4" class="svelte-m6d381"></path><path d="M255.923 140.977L0.236328 280.68V362.294L255.923 222.591V140.977Z" fill="#FF7C00" class="svelte-m6d381"></path></g></svg></div>');
function cu(e, t) {
  jr(t, !0);
  const r = () => fi(h, "$top", i), n = () => fi(c, "$bottom", i), [i, s] = Vs();
  var u = this && this.__awaiter || function(T, y, S, B) {
    function N(k) {
      return k instanceof S ? k : new S(function(U) {
        U(k);
      });
    }
    return new (S || (S = Promise))(function(k, U) {
      function R(P) {
        try {
          D(B.next(P));
        } catch (I) {
          U(I);
        }
      }
      function $(P) {
        try {
          D(B.throw(P));
        } catch (I) {
          U(I);
        }
      }
      function D(P) {
        P.done ? k(P.value) : N(P.value).then(R, $);
      }
      D((B = B.apply(T, y || [])).next());
    });
  };
  let f = C(t, "margin", 3, !0);
  const h = Ci([0, 0]), c = Ci([0, 0]);
  let d = Y(!1);
  function _() {
    return u(this, void 0, void 0, function* () {
      yield Promise.all([h.set([125, 140]), c.set([-125, -140])]), yield Promise.all([h.set([-125, 140]), c.set([125, -140])]), yield Promise.all([h.set([-125, 0]), c.set([125, -0])]), yield Promise.all([h.set([125, 0]), c.set([-125, 0])]);
    });
  }
  function b() {
    return u(this, void 0, void 0, function* () {
      yield _(), o(d) || b();
    });
  }
  function H() {
    return u(this, void 0, void 0, function* () {
      yield Promise.all([h.set([125, 0]), c.set([-125, 0])]), b();
    });
  }
  We(() => (H(), () => {
    x(d, !0);
  }));
  var O = fu();
  let w;
  var m = te(O), v = te(m), E = z(v);
  oe(() => {
    w = ht(O, 1, "svelte-m6d381", null, w, { margin: f() }), qe(v, `transform: translate(${r()[0] ?? ""}px, ${r()[1] ?? ""}px);`), qe(E, `transform: translate(${n()[0] ?? ""}px, ${n()[1] ?? ""}px);`);
  }), j(e, O), Ur(), s();
}
var hu = function(e, t, r, n) {
  function i(s) {
    return s instanceof r ? s : new r(function(u) {
      u(s);
    });
  }
  return new (r || (r = Promise))(function(s, u) {
    function f(d) {
      try {
        c(n.next(d));
      } catch (_) {
        u(_);
      }
    }
    function h(d) {
      try {
        c(n.throw(d));
      } catch (_) {
        u(_);
      }
    }
    function c(d) {
      d.done ? s(d.value) : i(d.value).then(f, h);
    }
    c((n = n.apply(e, t || [])).next());
  });
};
let Pr = [], mn = !1;
const du = typeof window < "u", Ha = du ? window.requestAnimationFrame : (e) => {
};
function pu(e) {
  return hu(this, arguments, void 0, function* (t, r = !0) {
    if (!(window.__gradio_mode__ === "website" || window.__gradio_mode__ !== "app" && r !== !0)) {
      if (Pr.push(t), !mn) mn = !0;
      else return;
      yield Ds(), Ha(() => {
        let n = [0, 0];
        for (let i = 0; i < Pr.length; i++) {
          const u = Pr[i].getBoundingClientRect();
          (i === 0 || u.top + window.scrollY <= n[0]) && (n[0] = u.top + window.scrollY, n[1] = i);
        }
        window.scrollTo({ top: n[0] - 20, behavior: "smooth" }), mn = !1, Pr = [];
      });
    }
  });
}
var vu = /* @__PURE__ */ Ee('<div class="validation-error svelte-124hqw6"> <button class="svelte-124hqw6"><!></button></div>'), mu = /* @__PURE__ */ Ee('<div class="eta-bar svelte-124hqw6"></div>'), gu = /* @__PURE__ */ Ee("<!> ", 1), bu = /* @__PURE__ */ Ee("<!> <!> <!> <!>", 1), _u = /* @__PURE__ */ Ee('<div class="progress-level svelte-124hqw6"><div class="progress-level-inner svelte-124hqw6"><!></div> <div class="progress-bar-wrap svelte-124hqw6"><div class="progress-bar svelte-124hqw6"></div></div></div>'), yu = /* @__PURE__ */ Ee('<p class="loading svelte-124hqw6"> </p> <!>', 1), xu = /* @__PURE__ */ Ee("<!> <div><!> <!></div> <!> <!>", 1), Eu = /* @__PURE__ */ Ee('<div class="clear-status svelte-124hqw6"><!></div> <span class="error svelte-124hqw6"> </span> <!>', 1), wu = /* @__PURE__ */ Ee("<div> <!> </div>"), Tu = /* @__PURE__ */ Ee('<div data-testid="status-tracker"><!> <!></div> <!>', 1);
function Su(e, t) {
  jr(t, !0);
  let r = C(t, "eta", 3, null), n = C(t, "scroll_to_output", 3, !1), i = C(t, "timer", 3, !0), s = C(t, "show_progress", 3, "full"), u = C(t, "message", 3, null), f = C(t, "progress", 3, null), h = C(t, "variant", 3, "default"), c = C(t, "loading_text", 3, "Loading..."), d = C(t, "absolute", 3, !0), _ = C(t, "translucent", 3, !1), b = C(t, "border", 3, !1), H = C(t, "validation_error", 7, null), O = C(t, "show_validation_error", 3, !0), w = C(t, "type", 3, null), m = C(t, "used_cache", 3, null), v = C(t, "cache_duration", 3, null), E = C(t, "avg_time", 3, null), T, y = !1, S = Y(0), B = Y(null), N = Y(null), k = Y(!1), U = Y(null), R = Y(!1), $ = Y(!1), D = Y(null), P = Y(null), I = Y("from cache"), le = Y(!1), ue = null, be = null;
  const Le = Xe(() => !(O() && H()) && (w() === "input" || !t.status || t.status === "complete" || s() === "hidden" || t.status == "streaming"));
  let Ze = Y(0);
  const dt = Xe(() => o(N) === null || o(N) <= 0 || !o(Ze) ? 0 : Math.min(o(Ze) / o(N), 1)), he = Xe(() => o(Ze).toFixed(1));
  let Ge = Xe(() => f() == null), ne = Xe(() => r() !== null && r() !== void 0 ? r() : o(B));
  function Ye() {
    Ha(() => {
      x(Ze, (performance.now() - o(S)) / 1e3), y && Ye();
    });
  }
  let fe = Xe(() => {
    let W = null;
    f() != null ? W = f().map((pe) => {
      if (pe.index != null && pe.length != null)
        return pe.index / pe.length;
      if (pe.progress != null)
        return pe.progress;
    }) : W = null;
    let de, Q = "";
    return W ? (de = W[W.length - 1], de === 0 ? Q = "0" : Q = "150ms") : de = void 0, {
      progress_level: W,
      last_progress_level: de,
      progress_bar_transition: Q
    };
  });
  function _e() {
    y || (x(B, x(U, null), !0), x(S, performance.now(), !0), y = !0, Ye());
  }
  function tt() {
    x(B, x(U, null), !0), y && (y = !1);
  }
  We(() => {
    t.status === "pending" ? _e() : xe(() => {
      tt();
    });
  }), We(() => {
    T && n() && (t.status === "pending" || t.status === "complete") && pu(T, t.autoscroll);
  }), We(() => {
    o(ne) != null && o(B) !== o(ne) && (x(N, (performance.now() - o(S)) / 1e3 + o(ne)), x(U, o(N).toFixed(1), !0), x(B, o(ne), !0));
  });
  function Ce() {
    x(k, !1);
  }
  We(() => {
    xe(() => {
      Ce();
    }), t.status === "error" && u() && x(k, !0);
  }), We(() => {
    t.status === "complete" && w() === "output" && m() && v() != null && (x(D, v().toFixed(1), !0), x(I, m() === "full" ? "from cache" : "used cache", !0), x(le, E() != null && E() > v() && E() > 0, !0), x(P, o(le) ? E().toFixed(1) : null, !0), x(R, !0), x($, !1), ue && clearTimeout(ue), be && clearTimeout(be), ue = setTimeout(
      () => {
        x($, !0), be = setTimeout(
          () => {
            x(R, !1), x($, !1);
          },
          500
        );
      },
      1750
    ));
  });
  var ut = Tu(), Je = Ne(ut);
  let Re, ye;
  var Fe = te(Je);
  {
    var It = (W) => {
      var de = vu(), Q = te(de), pe = z(Q), ce = te(pe);
      {
        let Ae = Xe(() => t.i18n ? t.i18n("common.clear") : "Clear");
        Ii(ce, {
          get Icon() {
            return Li;
          },
          get label() {
            return o(Ae);
          },
          disabled: !1,
          size: "x-small",
          background: "var(--background-fill-primary)",
          color: "var(--error-background-text)",
          border: "var(--border-color-primary)",
          onclick: () => H(null)
        });
      }
      oe(() => ge(Q, `${H() ?? ""} `)), j(W, de);
    };
    se(Fe, (W) => {
      H() && O() && W(It);
    });
  }
  var Tt = z(Fe, 2);
  {
    var Jt = (W) => {
      var de = xu(), Q = Ne(de);
      {
        var pe = (J) => {
          var ae = mu();
          let Te;
          oe(() => Te = qe(ae, "", Te, {
            transform: `translateX(${(o(dt) || 0) * 100 - 100}%)`
          })), j(J, ae);
        };
        se(Q, (J) => {
          h() === "default" && o(Ge) && s() === "full" && J(pe);
        });
      }
      var ce = z(Q, 2);
      let Ae;
      var q = te(ce);
      {
        var rt = (J) => {
          var ae = Ut(), Te = Ne(ae);
          wn(Te, 17, f, xn, (St, Ue) => {
            var nt = Ut(), Kt = Ne(nt);
            {
              var At = (ke) => {
                var Ht = gu(), Lt = Ne(Ht);
                {
                  var Bt = (je) => {
                    var it = ot();
                    oe((vt, mt) => ge(it, `${vt ?? ""}/${mt ?? ""}`), [
                      () => vn(o(Ue).index || 0),
                      () => vn(o(Ue).length)
                    ]), j(je, it);
                  }, pt = (je) => {
                    var it = ot();
                    oe((vt) => ge(it, vt), [() => vn(o(Ue).index || 0)]), j(je, it);
                  };
                  se(Lt, (je) => {
                    o(Ue).length != null ? je(Bt) : je(pt, -1);
                  });
                }
                var Qe = z(Lt);
                oe(() => ge(Qe, ` ${o(Ue).unit ?? ""} |  `)), j(ke, Ht);
              };
              se(Kt, (ke) => {
                o(Ue).index != null && ke(At);
              });
            }
            j(St, nt);
          }), j(J, ae);
        }, we = (J) => {
          var ae = ot();
          oe(() => ge(ae, `queue: ${t.queue_position + 1}/${t.queue_size ?? ""} |`)), j(J, ae);
        }, ve = (J) => {
          var ae = ot("processing |");
          j(J, ae);
        };
        se(q, (J) => {
          f() ? J(rt) : t.queue_position !== null && t.queue_size !== void 0 && t.queue_position >= 0 ? J(we, 1) : t.queue_position === 0 && J(ve, 2);
        });
      }
      var Be = z(q, 2);
      {
        var Yr = (J) => {
          var ae = ot();
          oe(() => ge(ae, `${o(he) ?? ""}${r() ? `/${o(U)}` : ""}s`)), j(J, ae);
        };
        se(Be, (J) => {
          i() && J(Yr);
        });
      }
      var dr = z(ce, 2);
      {
        var pr = (J) => {
          var ae = _u(), Te = te(ae), St = te(Te);
          {
            var Ue = (ke) => {
              var Ht = Ut(), Lt = Ne(Ht);
              wn(Lt, 17, f, xn, (Bt, pt, Qe) => {
                var je = Ut(), it = Ne(je);
                {
                  var vt = (mt) => {
                    var Ct = bu(), ft = Ne(Ct);
                    {
                      var Rt = (me) => {
                        var Pe = ot(" /");
                        j(me, Pe);
                      };
                      se(ft, (me) => {
                        Qe !== 0 && me(Rt);
                      });
                    }
                    var ct = z(ft, 2);
                    {
                      var gr = (me) => {
                        var Pe = ot();
                        oe(() => ge(Pe, o(pt).desc)), j(me, Pe);
                      };
                      se(ct, (me) => {
                        o(pt).desc != null && me(gr);
                      });
                    }
                    var gt = z(ct, 2);
                    {
                      var Me = (me) => {
                        var Pe = ot("-");
                        j(me, Pe);
                      };
                      se(gt, (me) => {
                        o(pt).desc != null && o(fe).progress_level && o(fe).progress_level[Qe] != null && me(Me);
                      });
                    }
                    var br = z(gt, 2);
                    {
                      var _r = (me) => {
                        var Pe = ot();
                        oe((yr) => ge(Pe, `${yr ?? ""}%`), [
                          () => (100 * (o(fe).progress_level[Qe] || 0)).toFixed(1)
                        ]), j(me, Pe);
                      };
                      se(br, (me) => {
                        o(fe).progress_level != null && me(_r);
                      });
                    }
                    j(mt, Ct);
                  };
                  se(it, (mt) => {
                    (o(pt).desc != null || o(fe).progress_level && o(fe).progress_level[Qe] != null) && mt(vt);
                  });
                }
                j(Bt, je);
              }), j(ke, Ht);
            };
            se(St, (ke) => {
              f() != null && ke(Ue);
            });
          }
          var nt = z(Te, 2), Kt = te(nt);
          let At;
          oe(() => At = qe(Kt, "", At, {
            width: `${o(fe).last_progress_level * 100}%`,
            transition: o(fe).progress_bar_transition
          })), j(J, ae);
        }, vr = (J) => {
          {
            let ae = Xe(() => h() === "default");
            cu(J, {
              get margin() {
                return o(ae);
              }
            });
          }
        };
        se(dr, (J) => {
          o(fe).last_progress_level != null ? J(pr) : s() === "full" && J(vr, 1);
        });
      }
      var Qt = z(dr, 2);
      {
        var mr = (J) => {
          var ae = yu(), Te = Ne(ae), St = te(Te), Ue = z(Te, 2);
          Tn(Ue, t, "additional-loading-text", {}), oe(() => ge(St, c())), j(J, ae);
        };
        se(Qt, (J) => {
          i() || J(mr);
        });
      }
      oe(() => Ae = ht(ce, 1, "progress-text svelte-124hqw6", null, Ae, {
        "meta-text-center": h() === "center",
        "meta-text": h() === "default"
      })), j(W, de);
    }, hr = (W) => {
      var de = Eu(), Q = Ne(de), pe = te(Q);
      {
        let rt = Xe(() => t.i18n("common.clear"));
        Ii(pe, {
          get Icon() {
            return Li;
          },
          get label() {
            return o(rt);
          },
          disabled: !1,
          $$events: {
            click: () => {
              t.on_clear_status?.();
            }
          }
        });
      }
      var ce = z(Q, 2), Ae = te(ce), q = z(ce, 2);
      Tn(q, t, "error", {}), oe((rt) => ge(Ae, rt), [() => t.i18n("common.error")]), j(W, de);
    };
    se(Tt, (W) => {
      t.status === "pending" ? W(Jt) : t.status === "error" && W(hr, 1);
    });
  }
  sr(Je, (W) => T = W, () => T);
  var Zr = z(Je, 2);
  {
    var He = (W) => {
      var de = wu();
      let Q, pe;
      var ce = te(de), Ae = z(ce);
      {
        var q = (we) => {
          var ve = ot();
          oe(() => ge(ve, `~${o(P) ?? ""}s
			→ `)), j(we, ve);
        };
        se(Ae, (we) => {
          o(le) && we(q);
        });
      }
      var rt = z(Ae);
      oe(() => {
        Q = ht(de, 1, "cache-indicator svelte-124hqw6", null, Q, { "fade-out": o($) }), pe = qe(de, "", pe, { position: d() ? "absolute" : "static" }), ge(ce, `⚡ ${o(I) ?? ""}: `), ge(rt, `${o(D) ?? ""}s`);
      }), j(W, de);
    };
    se(Zr, (W) => {
      o(R) && W(He);
    });
  }
  oe(() => {
    Re = ht(Je, 1, `wrap ${h() ?? ""} ${s() ?? ""}`, "svelte-124hqw6", Re, {
      "no-click": H() && O(),
      hide: o(Le),
      translucent: h() === "center" && (t.status === "pending" || t.status === "error") || _() || s() === "minimal" || H(),
      generating: t.status === "generating" && s() === "full",
      border: b()
    }), ye = qe(Je, "", ye, {
      position: d() ? "absolute" : "static",
      padding: d() ? "0" : "var(--size-8) 0"
    });
  }), j(e, ut), Ur();
}
const Au = (e) => {
  const t = {};
  for (let r = 0, n = e.length; r < n; r++) {
    const i = e[r];
    for (const s in i)
      t[s] ? t[s] = t[s].concat(i[s]) : t[s] = i[s];
  }
  return t;
}, Hu = [
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
], Mu = [
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
], Pu = [
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
Au([
  Object.fromEntries(Hu.map((e) => [e, ["*"]])),
  Object.fromEntries(Mu.map((e) => [e, ["svg:*"]])),
  Object.fromEntries(Pu.map((e) => [e, ["math:*"]]))
]);
Fr(["touchstart", "touchmove", "touchend", "click", "keydown"]);
var Ou = /* @__PURE__ */ new Set(["$$slots", "$$events", "$$legacy"]), Nu = /* @__PURE__ */ Ee("<option> </option>"), Iu = /* @__PURE__ */ Ee('<div class="group-picker svelte-r41nsf"><label for="layout-active-label" class="svelte-r41nsf">当前 Label</label> <select id="layout-active-label" class="svelte-r41nsf"></select></div>'), Lu = /* @__PURE__ */ Ee('<!> <div class="layout-editor svelte-r41nsf"><!> <div class="toolbar svelte-r41nsf"><button type="button" class="svelte-r41nsf">居中归一</button> <button type="button" class="svelte-r41nsf">居中</button> <button type="button" class="svelte-r41nsf">适配</button> <button type="button" class="svelte-r41nsf">找回视野</button> <button type="button">差异混合</button></div> <div class="canvas-wrap svelte-r41nsf"><div class="hud svelte-r41nsf" aria-live="polite"><div> </div> <div> </div> <div> </div></div> <div><canvas class="svelte-r41nsf"></canvas></div> <canvas tabindex="0" role="application" aria-label="版图变换编辑器" class="svelte-r41nsf"></canvas></div> <div class="shortcut-bar svelte-r41nsf">拖拽 0.25× · Shift 拖拽 1× · 方向键/WASD 微调 · Shift 10px · [ ] 旋转 · Esc 取消 · Ctrl+Z 撤销 · 空格+拖 平移 · 滚轮/+/- 缩放</div> <div class="status svelte-r41nsf"> </div></div>', 1);
function Cu(e, t) {
  jr(t, !0);
  const r = /* @__PURE__ */ yo(t, Ou), n = 2048, i = 35e-5, s = 4, u = 0.25, f = 1, h = 0.18, c = 140, d = 3, _ = 1, b = 10, H = 0.5, O = [
    [0, 255, 120],
    [0, 188, 255],
    [255, 179, 0],
    [213, 94, 255],
    [255, 82, 82],
    [75, 222, 196]
  ], w = new nu(r);
  let m, v, E, T = null, y = null, S = null, B = null, N = /* @__PURE__ */ new Map(), k = /* @__PURE__ */ new Map(), U = Y(!1), R = Y(!1), $ = Y("等待版图 mask"), D = Y(or({ enabled: !1 })), P = Y(or(hr())), I = Y(""), le = Y(!1), ue = Y(!1), be = Y(!1), Le = Y(!1), Ze = Y(!1), dt = Y(!1), he = Y(!1), Ge = Y(u), ne = Y("crosshair"), Ye = "", fe = 0, _e = { x: 0, y: 0, clientX: 0, clientY: 0, center_x: 0, center_y: 0 }, tt = { angle: 0, rotation: 0 }, Ce = { x: 0, y: 0, scrollLeft: 0, scrollTop: 0 }, ut = { x: 0, y: 0 }, Je = { x: 0, y: 0 }, Re = null, ye = null, Fe = [], It = !1, Tt = null, Jt = /* @__PURE__ */ new Set();
  function hr() {
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
  function Zr(a) {
    return JSON.parse(JSON.stringify(a || { enabled: !1 }));
  }
  function He(a) {
    const l = Object.assign(Object.assign({}, hr()), a || {});
    return Object.assign(Object.assign({}, l), {
      center_x: W(l.center_x, 0),
      center_y: W(l.center_y, 0),
      pivot_x: W(l.pivot_x, 0),
      pivot_y: W(l.pivot_y, 0),
      scale: Q(W(l.scale, 1), 0.01, 20),
      rotation_deg: pe(W(l.rotation_deg, 0)),
      preview_alpha: Q(W(l.preview_alpha, 0.35), 0, 1),
      revision: Math.max(0, Math.trunc(W(l.revision, 0)))
    });
  }
  function W(a, l) {
    const p = Number(a);
    return Number.isFinite(p) ? p : l;
  }
  function de(a) {
    return typeof a == "number" ? String(a) + "px" : a || "520px";
  }
  function Q(a, l, p) {
    return Math.max(l, Math.min(p, a));
  }
  function pe(a) {
    let l = ((a + 180) % 360 + 360) % 360 - 180;
    return l === -180 && (l = 180), l;
  }
  function ce(a) {
    return typeof a + ":" + String(a);
  }
  function Ae() {
    var a;
    return q() ? Array.isArray((a = o(D).group_view) === null || a === void 0 ? void 0 : a.groups) ? o(D).group_view.groups : [] : [];
  }
  function q() {
    return o(D).transform_mode === "label_groups" && !!o(D).group_view;
  }
  function rt() {
    for (const a of Ae())
      if (ce(a.group_id) === o(I)) return a;
    return null;
  }
  function we() {
    const a = rt();
    return a ? String(a.label || "Label " + String(a.group_id)) : "";
  }
  function ve() {
    return Math.max(1, Number(o(D).target_width || T?.naturalWidth || 1));
  }
  function Be() {
    return Math.max(1, Number(o(D).target_height || T?.naturalHeight || 1));
  }
  function Yr() {
    var a;
    const l = (a = N.get(o(I))) === null || a === void 0 ? void 0 : a.image;
    return Math.max(1, Number(o(D).source_width || y?.naturalWidth || l?.naturalWidth || 1));
  }
  function dr() {
    var a;
    const l = (a = N.get(o(I))) === null || a === void 0 ? void 0 : a.image;
    return Math.max(1, Number(o(D).source_height || y?.naturalHeight || l?.naturalHeight || 1));
  }
  function pr() {
    return Math.min(1, n / Math.max(ve(), Be()));
  }
  function vr(a = o(I)) {
    var l;
    if (q()) {
      const g = (l = N.get(a)) === null || l === void 0 ? void 0 : l.view.foreground_bbox_xyxy;
      if (Array.isArray(g) && g.length >= 4) return g.slice(0, 4).map(Number);
    }
    const p = o(D).foreground_bbox_xyxy;
    return Array.isArray(p) && p.length >= 4 ? p.slice(0, 4).map(Number) : [0, 0, Yr() - 1, dr() - 1];
  }
  function Qt(a, l, p) {
    if (!a) {
      l === fe && p(null);
      return;
    }
    const g = new Image();
    g.onload = () => {
      l === fe && p(g);
    }, g.onerror = () => {
      l === fe && p(null);
    }, g.src = a;
  }
  function mr(a, l) {
    const p = Math.max(1, Number(o(D).source_width || a.naturalWidth || 1)), g = Math.max(1, Number(o(D).source_height || a.naturalHeight || 1)), M = document.createElement("canvas");
    M.width = p, M.height = g;
    const A = M.getContext("2d", { willReadFrequently: !0 });
    if (!A) return null;
    A.imageSmoothingEnabled = !1, A.drawImage(a, 0, 0, p, g);
    const L = A.getImageData(0, 0, p, g), G = document.createElement("canvas");
    G.width = p, G.height = g;
    const F = G.getContext("2d");
    if (!F) return null;
    const K = F.createImageData(p, g);
    for (let ee = 0; ee < L.data.length; ee += 4) {
      const re = Math.max(L.data[ee], L.data[ee + 1], L.data[ee + 2]);
      L.data[ee + 3] > 0 && re >= 128 && (K.data[ee] = l[0], K.data[ee + 1] = l[1], K.data[ee + 2] = l[2], K.data[ee + 3] = 255);
    }
    return F.putImageData(K, 0, 0), { mask: M, tint: G };
  }
  function J() {
    if (!y) {
      S = null, B = null;
      return;
    }
    const a = mr(y, O[0]);
    S = a?.mask || null, B = a?.tint || null;
  }
  function ae(a) {
    if (!a.image) {
      a.maskCanvas = null, a.tintCanvas = null, a.ready = !1;
      return;
    }
    const l = mr(a.image, a.color);
    a.maskCanvas = l?.mask || null, a.tintCanvas = l?.tint || null, a.ready = !!l;
  }
  function Te() {
    Tt && (clearTimeout(Tt), Tt = null);
  }
  function St(a) {
    const l = o(D).group_view, p = Array.isArray(l?.groups) ? l.groups : [], g = o(D).group_intent, M = !!l && !!g && String(g.selection_signature || "") === String(l.selection_signature || ""), A = /* @__PURE__ */ new Map();
    if (M && Array.isArray(g?.transforms))
      for (const re of g.transforms)
        !re || re.group_id === void 0 || !re.transform || A.set(ce(re.group_id), He(re.transform));
    const L = He(o(D).transform), G = /* @__PURE__ */ new Map(), F = /* @__PURE__ */ new Map(), K = /* @__PURE__ */ new Set();
    for (let re = 0; re < p.length; re++) {
      const Ve = p[re];
      if (!Ve || Ve.group_id === void 0 || Ve.group_id === null) continue;
      const Ke = ce(Ve.group_id);
      if (K.has(Ke)) continue;
      K.add(Ke);
      const at = {
        view: Ve,
        image: null,
        maskCanvas: null,
        tintCanvas: null,
        ready: !1,
        color: O[re % O.length]
      };
      G.set(Ke, at), F.set(Ke, He(A.get(Ke) || L));
    }
    N = G, k = F, Jt = /* @__PURE__ */ new Set();
    const ee = M && g?.active_group_id !== void 0 && g.active_group_id !== null ? ce(g.active_group_id) : "";
    x(
      I,
      G.has(ee) ? ee : G.keys().next().value || "",
      !0
    ), x(P, He(k.get(o(I)) || L), !0), o(I) && k.set(o(I), o(P)), y = null, S = null, B = null, x(R, !1);
    for (const [re, Ve] of G)
      Qt(Ve.view.mask_image, a, (Ke) => {
        const at = N.get(re);
        !at || at !== Ve || (at.image = Ke, ae(at), De());
      });
    x(
      $,
      o(D).status || (p.length > 0 ? "已加载 " + String(p.length) + " 个 Label；当前：" + we() : "当前选择没有可编辑 Label"),
      !0
    );
  }
  function Ue(a) {
    Te(), fe += 1;
    const l = fe;
    x(D, Zr(a), !0), x(U, !1), N = /* @__PURE__ */ new Map(), k = /* @__PURE__ */ new Map(), x(I, ""), x(le, !1), x(ue, !1), x(be, !1), x(Le, !1), Re = null, ye = null, Qt(o(D).base_image, l, (p) => {
      T = p, x(U, !!p), De();
    }), q() ? St(l) : (x(P, He(o(D).transform), !0), x($, o(D).status || "编辑器已加载", !0), x(R, !1), Qt(o(D).mask_image, l, (p) => {
      y = p, x(R, !!p), J(), De();
    }));
  }
  We(() => {
    const a = JSON.stringify(w.props.value || null);
    a !== Ye && (Ye = a, Ue(w.props.value));
  }), Qs(() => {
    fe += 1, Te(), window.removeEventListener("keydown", vt), window.removeEventListener("keyup", mt);
  });
  function nt() {
    if (!m) return !1;
    const a = document.activeElement;
    return a === m || m.contains(a);
  }
  function Kt() {
    return o(Ze) ? "键盘已接管" : o($);
  }
  function At(a) {
    return a.shiftKey ? f : u;
  }
  function ke() {
    const a = Object.assign({}, He(o(P)));
    Fe = [...Fe.slice(-29), a];
  }
  function Ht() {
    if (!Se() || Fe.length === 0) return;
    const a = Fe.pop();
    a && (Me(a), $t("undo", "已撤销上一步变换"));
  }
  function Lt() {
    var a;
    if (!(!o(le) && !o(be) && !o(ue))) {
      ye && Me(ye), x(le, !1), x(ue, !1), x(be, !1), x(Le, !1), Re = null, ye = null, x(ne, "crosshair");
      try {
        !((a = m?.hasPointerCapture) === null || a === void 0) && a.call(m, 0) && m.releasePointerCapture(0);
      } catch {
      }
      De();
    }
  }
  function Bt() {
    ke(), ye = Object.assign({}, He(o(P)));
  }
  function pt(a) {
    var l;
    const p = Q(Number((l = a.preview_alpha) !== null && l !== void 0 ? l : 0.35), 0, 1);
    return o(le) || o(be) ? Math.min(p, h) : p;
  }
  function Qe(a, l) {
    !Se() || !nt() || (ke(), Me(Object.assign(Object.assign({}, o(P)), {
      center_x: Number(o(P).center_x || 0) + a,
      center_y: Number(o(P).center_y || 0) + l
    })), $t("keyboard", "键盘微调已同步"));
  }
  function je(a) {
    !Se() || !nt() || (ke(), Me(Object.assign(Object.assign({}, o(P)), {
      rotation_deg: pe(Number(o(P).rotation_deg || 0) + a)
    })), $t("keyboard", "键盘旋转已同步"));
  }
  function it(a) {
    if (!Se() || !nt()) return;
    ke();
    const l = ve() / 2, p = Be() / 2, g = Rt(l, p, o(P)), M = Q(Number(o(P).scale || 1) * a, 0.01, 20);
    let A = Object.assign(Object.assign({}, o(P)), { scale: M });
    const L = ft(g.x, g.y, A);
    A = Object.assign(Object.assign({}, A), {
      center_x: Number(A.center_x || 0) + l - L.x,
      center_y: Number(A.center_y || 0) + p - L.y
    }), Me(A), $t("keyboard", "键盘缩放已同步: scale=" + M.toFixed(3));
  }
  function vt(a) {
    if (a.code === "Space") {
      It = !0, nt() && a.preventDefault();
      return;
    }
    if (!nt()) return;
    const l = a.key;
    if (l === "Escape") {
      a.preventDefault(), Lt();
      return;
    }
    if ((a.ctrlKey || a.metaKey) && l.toLowerCase() === "z" && !a.shiftKey) {
      a.preventDefault(), Ht();
      return;
    }
    if (!Se()) return;
    const p = a.shiftKey ? b : _;
    l === "ArrowLeft" || l === "a" || l === "A" ? (a.preventDefault(), Qe(-p, 0)) : l === "ArrowRight" || l === "d" || l === "D" ? (a.preventDefault(), Qe(p, 0)) : l === "ArrowUp" || l === "w" || l === "W" ? (a.preventDefault(), Qe(0, -p)) : l === "ArrowDown" || l === "s" || l === "S" ? (a.preventDefault(), Qe(0, p)) : l === "[" ? (a.preventDefault(), je(-H)) : l === "]" ? (a.preventDefault(), je(H)) : l === "+" || l === "=" ? (a.preventDefault(), it(1.08)) : (l === "-" || l === "_") && (a.preventDefault(), it(1 / 1.08));
  }
  function mt(a) {
    a.code === "Space" && (It = !1);
  }
  typeof window < "u" && (window.addEventListener("keydown", vt), window.addEventListener("keyup", mt));
  function Ct(a) {
    const l = Number(a.rotation_deg || 0) * Math.PI / 180, p = Number(a.scale || 1), g = Math.cos(l), M = Math.sin(l), A = p * g, L = p * M, G = Number(a.center_x || 0) - A * Number(a.pivot_x || 0) + L * Number(a.pivot_y || 0), F = Number(a.center_y || 0) - L * Number(a.pivot_x || 0) - A * Number(a.pivot_y || 0);
    return [A, L, -L, A, G, F];
  }
  function ft(a, l, p = o(P)) {
    const [g, M, A, L, G, F] = Ct(p);
    return { x: g * a + A * l + G, y: M * a + L * l + F };
  }
  function Rt(a, l, p = o(P)) {
    const [g, M, A, L, G, F] = Ct(p), K = g * L - M * A;
    if (Math.abs(K) < 1e-9) return { x: -1, y: -1 };
    const ee = a - G, re = l - F;
    return { x: (L * ee - A * re) / K, y: (-M * ee + g * re) / K };
  }
  function ct(a) {
    const l = m.getBoundingClientRect();
    return {
      x: (a.clientX - l.left) / Math.max(1, l.width) * ve(),
      y: (a.clientY - l.top) / Math.max(1, l.height) * Be()
    };
  }
  function gr(a, l, p) {
    if (!a) return !1;
    const g = Math.round(l), M = Math.round(p);
    if (g < 0 || M < 0 || g >= a.width || M >= a.height) return !1;
    const A = a.getContext("2d", { willReadFrequently: !0 });
    if (!A) return !1;
    const L = A.getImageData(g, M, 1, 1).data;
    return L[3] > 0 && Math.max(L[0], L[1], L[2]) >= 128;
  }
  function gt(a) {
    return !q() || a === o(I) ? o(P) : k.get(a) || o(P);
  }
  function Me(a) {
    x(P, He(a), !0), q() && o(I) && k.set(o(I), o(P));
  }
  function br(a, l) {
    const p = Rt(a, l, o(P));
    return gr(S, p.x, p.y);
  }
  function _r(a, l, p) {
    const g = N.get(a);
    if (!g?.ready) return !1;
    const M = Rt(l, p, gt(a));
    return gr(g.maskCanvas, M.x, M.y);
  }
  function me(a, l) {
    if (o(I) && _r(o(I), a, l)) return o(I);
    const p = Ae().map((g) => ce(g.group_id)).reverse();
    for (const g of p)
      if (g !== o(I) && _r(g, a, l)) return g;
    return "";
  }
  function Pe(a = o(I), l = gt(a)) {
    const [p, g, M, A] = vr(a);
    return [
      ft(p, g, l),
      ft(M, g, l),
      ft(M, A, l),
      ft(p, A, l)
    ];
  }
  function yr() {
    const a = Pe();
    let l = a[0];
    for (const A of a)
      (A.y < l.y || Math.abs(A.y - l.y) < 1e-6 && A.x > l.x) && (l = A);
    const p = Number(o(P).rotation_deg || 0) * Math.PI / 180, g = Math.cos(p - Math.PI / 4), M = Math.sin(p - Math.PI / 4);
    return { x: l.x + g * 34, y: l.y + M * 34 };
  }
  function Xn() {
    const a = m?.getBoundingClientRect();
    return a ? Math.max(8, 14 * ve() / Math.max(1, a.width)) : 14;
  }
  function Wn(a, l) {
    if (q() && !o(I)) return !1;
    const p = yr(), g = Xn();
    return Math.hypot(a - p.x, l - p.y) <= g;
  }
  function qn(a) {
    const l = q() ? o(I) : "";
    Me(Object.assign(Object.assign({}, o(P)), {
      revision: Number(o(P).revision || 0) + 1,
      origin: a,
      scale: Q(Number(o(P).scale || 1), 0.01, 20),
      rotation_deg: pe(Number(o(P).rotation_deg || 0))
    })), l && Jt.add(l);
  }
  function Ma() {
    const a = [];
    for (const l of Ae()) {
      const p = ce(l.group_id);
      a.push({
        group_id: l.group_id,
        transform: Object.assign({}, He(gt(p)))
      });
    }
    return a;
  }
  function Zn() {
    const a = {
      enabled: o(D).enabled,
      transform: Object.assign({}, He(o(P))),
      target_width: o(D).target_width,
      target_height: o(D).target_height
    };
    q() && (a.transform_mode = "label_groups", a.group_intent = o(D).group_intent ? JSON.parse(JSON.stringify(o(D).group_intent)) : null), w.props.value = a, Ye = JSON.stringify(a);
  }
  function Yn(a, l, p, g = !0) {
    var M, A;
    if (!q()) return;
    p && qn(a);
    const L = o(D).group_view, G = rt(), F = Ae().filter((ee) => Jt.has(ce(ee.group_id))).map((ee) => ee.group_id), K = Number(((M = o(D).group_intent) === null || M === void 0 ? void 0 : M.transform_set_revision) || 0);
    x(
      D,
      Object.assign(Object.assign({}, o(D)), {
        group_intent: {
          selection_signature: String(L?.selection_signature || ""),
          transform_set_revision: K + 1,
          active_group_id: (A = G?.group_id) !== null && A !== void 0 ? A : null,
          changed_group_ids: F,
          transforms: Ma()
        }
      }),
      !0
    ), x($, l, !0), Zn(), g && (Te(), w.dispatch("change")), De();
  }
  function kt(a, l, p = !0) {
    if (q()) {
      Yn(a, l, !0, p);
      return;
    }
    qn(a), x(
      D,
      Object.assign(Object.assign({}, o(D)), {
        enabled: !0,
        transform: Object.assign({}, o(P)),
        status: l
      }),
      !0
    ), x($, l, !0), Zn(), p && (Te(), w.dispatch("change")), De();
  }
  function $t(a, l, p = 140) {
    kt(a, l, !1), Te(), Tt = setTimeout(
      () => {
        Tt = null, w.dispatch("change");
      },
      p
    );
  }
  function Pa(a, l, p) {
    if (T && o(U)) {
      a.imageSmoothingEnabled = !0, a.drawImage(T, 0, 0, l, p);
      return;
    }
    a.fillStyle = "#f8fafc", a.fillRect(0, 0, l, p), a.fillStyle = "#64748b", a.font = "18px sans-serif", a.fillText("请先上传图像", 24, 42);
  }
  function Jn(a, l, p, g) {
    a.save(), a.globalAlpha = pt(g), o(dt) && (a.globalCompositeOperation = "difference");
    const [M, A, L, G, F, K] = Ct(g);
    a.setTransform(l * M, l * A, l * L, l * G, l * F, l * K), a.imageSmoothingEnabled = !1, a.drawImage(p, 0, 0, p.width, p.height), a.restore();
  }
  function Qn(a, l) {
    if (!o(le) || !Re) return;
    const p = Pe(o(I), Re);
    a.save(), a.lineJoin = "round", a.setLineDash([Math.max(4, l / 180), Math.max(4, l / 180)]), a.lineWidth = Math.max(1.5, l / 900), a.strokeStyle = "rgba(255,255,255,0.72)", a.beginPath(), a.moveTo(p[0].x, p[0].y);
    for (let g = 1; g < p.length; g++) a.lineTo(p[g].x, p[g].y);
    a.closePath(), a.stroke(), a.restore();
  }
  function Kn(a, l) {
    const p = Pe();
    a.save(), a.lineJoin = "round", a.lineWidth = Math.max(2.5, l / 700), a.strokeStyle = "rgba(0,0,0,0.82)", a.beginPath(), a.moveTo(p[0].x, p[0].y);
    for (let A = 1; A < p.length; A++) a.lineTo(p[A].x, p[A].y);
    a.closePath(), a.stroke(), a.lineWidth = Math.max(1.8, l / 1e3), a.strokeStyle = "#00ff66", a.stroke();
    const g = yr(), M = p.reduce((A, L) => L.y < A.y || Math.abs(L.y - A.y) < 1e-6 && L.x > A.x ? L : A, p[0]);
    a.strokeStyle = "#0f172a", a.lineWidth = Math.max(2, l / 900), a.beginPath(), a.moveTo(M.x, M.y), a.lineTo(g.x, g.y), a.stroke(), a.fillStyle = o(be) ? "#ffb000" : "#ffffff", a.strokeStyle = "#00ff66", a.lineWidth = Math.max(2, l / 900), a.beginPath(), a.arc(g.x, g.y, Xn(), 0, Math.PI * 2), a.fill(), a.stroke(), a.restore();
  }
  function Oa(a, l, p) {
    const g = N.get(l);
    if (!g) return;
    const M = Pe(l, gt(l)), A = Math.min(...M.map((re) => re.x)), L = Math.min(...M.map((re) => re.y)), G = String(g.view.label || "Label " + String(g.view.group_id));
    a.save(), a.font = "600 13px sans-serif";
    const F = a.measureText(G).width + 12, K = Q(A, 2, Math.max(2, ve() - F - 2)), ee = Q(L - 23, 2, Math.max(2, Be() - 22));
    a.fillStyle = p ? "rgba(15,23,42,0.92)" : "rgba(51,65,85,0.76)", a.fillRect(K, ee, F, 20), a.fillStyle = "#ffffff", a.fillText(G, K + 6, ee + 14), a.restore();
  }
  function De() {
    var a, l;
    if (!m) return;
    const p = ve(), g = Be(), M = pr();
    m.width = Math.max(1, Math.round(p * M)), m.height = Math.max(1, Math.round(g * M));
    const A = m.getContext("2d");
    if (A && (A.setTransform(M, 0, 0, M, 0, 0), A.clearRect(0, 0, p, g), Pa(A, p, g), o(D).enabled !== !1)) {
      if (q()) {
        const G = Ae().map((F) => ce(F.group_id)).filter((F) => F !== o(I));
        o(I) && G.push(o(I));
        for (const F of G) {
          const K = N.get(F);
          K?.ready && K.tintCanvas && Jn(A, M, K.tintCanvas, gt(F));
        }
        A.setTransform(M, 0, 0, M, 0, 0);
        for (const F of G)
          !((a = N.get(F)) === null || a === void 0) && a.ready && Oa(A, F, F === o(I));
        o(I) && (!((l = N.get(o(I))) === null || l === void 0) && l.ready) && (Qn(A, p), Kn(A, p)), Jr();
        return;
      }
      B && o(R) && (Jn(A, M, B, o(P)), A.setTransform(M, 0, 0, M, 0, 0), Qn(A, p), Kn(A, p)), Jr();
    }
  }
  function Jr() {
    if (!o(he) || !m || !E || !o(U)) return;
    const a = pr(), l = ve(), p = Be(), g = E.getContext("2d");
    if (!g) return;
    E.width = c, E.height = c, g.clearRect(0, 0, c, c);
    const M = c / d, A = Q(ut.x * a - M / 2, 0, Math.max(0, l * a - M)), L = Q(ut.y * a - M / 2, 0, Math.max(0, p * a - M));
    g.imageSmoothingEnabled = !1, g.drawImage(m, A, L, M, M, 0, 0, c, c), g.strokeStyle = "#ffffff", g.lineWidth = 2, g.strokeRect(1, 1, c - 2, c - 2), g.strokeStyle = "rgba(15,23,42,0.55)", g.lineWidth = 1, g.beginPath(), g.moveTo(c / 2, 0), g.lineTo(c / 2, c), g.moveTo(0, c / 2), g.lineTo(c, c / 2), g.stroke();
  }
  function Se() {
    var a;
    return o(D).enabled ? q() ? !!o(I) && !!(!((a = N.get(o(I))) === null || a === void 0) && a.ready) : o(R) && !!S : !1;
  }
  function Qr(a) {
    return !q() || !N.has(a) || a === o(I) ? !1 : (o(I) && k.set(o(I), He(o(P))), x(I, a, !0), x(P, He(k.get(a)), !0), k.set(a, o(P)), x($, "当前 Label：" + we()), De(), !0);
  }
  function Na(a) {
    const l = a.currentTarget.value;
    Qr(l) && Yn("select_group", "已选择 Label：" + we(), !1, !0);
  }
  function Ia(a) {
    if (a.button !== 0) return;
    if (m?.focus(), !Se()) {
      x($, "请先启用并加载版图 mask"), De();
      return;
    }
    Te();
    const l = ct(a);
    if (It && v) {
      x(Le, !0), Ce = {
        x: a.clientX,
        y: a.clientY,
        scrollLeft: v.scrollLeft,
        scrollTop: v.scrollTop
      }, x(ne, "grab"), m.setPointerCapture(a.pointerId);
      return;
    }
    if (Wn(l.x, l.y)) {
      Bt(), Re = null, x(ne, "grabbing"), x(be, !0), tt = {
        angle: Math.atan2(l.y - o(P).center_y, l.x - o(P).center_x) * 180 / Math.PI,
        rotation: Number(o(P).rotation_deg || 0)
      }, m.setPointerCapture(a.pointerId);
      return;
    }
    if (q()) {
      const p = me(l.x, l.y);
      if (!p) {
        x($, "请点中任一 Label mask 前景后拖动"), De();
        return;
      }
      Qr(p);
    } else if (!br(l.x, l.y)) {
      x($, "请点中版图 mask 前景后拖动"), De();
      return;
    }
    Re = Object.assign({}, He(o(P))), x(ue, !0), _e = {
      x: l.x,
      y: l.y,
      clientX: a.clientX,
      clientY: a.clientY,
      center_x: Number(o(P).center_x || 0),
      center_y: Number(o(P).center_y || 0)
    }, x(ne, "grab"), m.setPointerCapture(a.pointerId);
  }
  function La(a, l) {
    if (ut = { x: l.x, y: l.y }, !v) return;
    const p = v.getBoundingClientRect(), g = 18;
    Je = {
      x: Q(a.clientX - p.left + g, 0, Math.max(0, p.width - c)),
      y: Q(a.clientY - p.top + g, 0, Math.max(0, p.height - c))
    }, x(he, !0);
  }
  function Ba(a) {
    !o(ue) || o(le) || Math.hypot(a.clientX - _e.clientX, a.clientY - _e.clientY) < s || (Bt(), x(le, !0), x(ue, !1), x(Ge, At(a), !0), x(ne, "grabbing"));
  }
  function xr(a) {
    if (o(Le)) {
      x(ne, "grabbing");
      return;
    }
    if (!Se()) {
      x(ne, "not-allowed");
      return;
    }
    const l = q() ? !!me(a.x, a.y) : br(a.x, a.y);
    if (It) {
      x(ne, "grab");
      return;
    }
    if (Wn(a.x, a.y) || l) {
      x(ne, "grab");
      return;
    }
    x(ne, "crosshair");
  }
  function Ca(a) {
    const l = ct(a);
    if (La(a, l), o(Le) && v) {
      v.scrollLeft = Ce.scrollLeft - (a.clientX - Ce.x), v.scrollTop = Ce.scrollTop - (a.clientY - Ce.y), x(ne, "grabbing");
      return;
    }
    if (o(ue) && Ba(a), !o(le) && !o(be) && !o(ue)) {
      xr(l), Jr();
      return;
    }
    if (o(le)) {
      x(Ge, At(a), !0);
      const p = o(Ge);
      Me(Object.assign(Object.assign({}, o(P)), {
        center_x: _e.center_x + (l.x - _e.x) * p,
        center_y: _e.center_y + (l.y - _e.y) * p
      })), x($, "正在拖动 " + (q() ? we() : "版图") + "；松开后同步变换");
    } else if (o(be)) {
      const p = Math.atan2(l.y - o(P).center_y, l.x - o(P).center_x) * 180 / Math.PI;
      Me(Object.assign(Object.assign({}, o(P)), {
        rotation_deg: pe(tt.rotation + p - tt.angle)
      })), x($, "正在旋转 " + (q() ? we() : "版图") + "；松开后同步变换");
    }
    De();
  }
  function Ra() {
    x(he, !1), !o(le) && !o(be) && !o(ue) && !o(Le) && x(ne, "crosshair");
  }
  function $n(a) {
    if (o(Le)) {
      x(Le, !1);
      try {
        m.releasePointerCapture(a.pointerId);
      } catch {
      }
      xr(ct(a));
      return;
    }
    if (o(ue) && !o(le)) {
      x(ue, !1), Re = null, ye = null;
      try {
        m.releasePointerCapture(a.pointerId);
      } catch {
      }
      xr(ct(a)), De();
      return;
    }
    if (o(le) || o(be)) {
      x(le, !1), x(ue, !1), x(be, !1), Re = null, ye = null;
      try {
        m.releasePointerCapture(a.pointerId);
      } catch {
      }
      xr(ct(a));
      const l = q() ? we() : "Canvas";
      kt("canvas", l + " 变换已同步；点击更新预览或创建实例后使用后端权威 mask");
    }
  }
  function ka(a) {
    if (!Se()) return;
    a.preventDefault(), ke();
    const l = ct(a);
    if (q()) {
      const G = me(l.x, l.y);
      G && Qr(G);
    }
    const p = Rt(l.x, l.y, o(P)), g = Math.exp(-a.deltaY * i), M = Q(Number(o(P).scale || 1) * g, 0.01, 20);
    let A = Object.assign(Object.assign({}, o(P)), { scale: M });
    const L = ft(p.x, p.y, A);
    A = Object.assign(Object.assign({}, A), {
      center_x: Number(A.center_x || 0) + l.x - L.x,
      center_y: Number(A.center_y || 0) + l.y - L.y
    }), Me(A), $t("canvas", "滚轮缩放已同步: scale=" + M.toFixed(3));
  }
  function Da() {
    Se() && (Me(Object.assign(Object.assign({}, o(P)), {
      center_x: ve() / 2,
      center_y: Be() / 2,
      scale: 1,
      rotation_deg: 0
    })), kt("reset", "居中归一 " + (q() ? we() : "版图") + "：scale=1，rotation=0"));
  }
  function Ga() {
    Se() && (Me(Object.assign(Object.assign({}, o(P)), { center_x: ve() / 2, center_y: Be() / 2 })), kt("center", "居中 " + (q() ? we() : "版图") + "：保留缩放和旋转"));
  }
  function Fa() {
    if (!Se()) return;
    const [a, l, p, g] = vr(), M = Math.max(1, p - a + 1), A = Math.max(1, g - l + 1), L = Q(Math.min(ve() / M, Be() / A) * 0.9, 0.01, 20);
    Me(Object.assign(Object.assign({}, o(P)), {
      center_x: ve() / 2,
      center_y: Be() / 2,
      scale: L
    })), kt("fit", "适配 " + (q() ? we() : "版图") + "：scale=" + L.toFixed(3));
  }
  function Ua() {
    if (!Se()) return;
    const a = Pe(), l = Math.min(...a.map((F) => F.x)), p = Math.max(...a.map((F) => F.x)), g = Math.min(...a.map((F) => F.y)), M = Math.max(...a.map((F) => F.y));
    let A = 0, L = 0;
    const G = Math.max(20, ve() * 0.03);
    if (p < G ? A = G - p : l > ve() - G && (A = ve() - G - l), M < G ? L = G - M : g > Be() - G && (L = Be() - G - g), A === 0 && L === 0) {
      x($, (q() ? we() : "版图") + " 已经在视野内");
      return;
    }
    Me(Object.assign(Object.assign({}, o(P)), {
      center_x: Number(o(P).center_x || 0) + A,
      center_y: Number(o(P).center_y || 0) + L
    })), kt("bring_into_view", "已找回 " + (q() ? we() : "版图") + " 到视野内");
  }
  {
    let a = Xe(() => o(le) || o(be) || o(Le) ? "focus" : "base");
    su(e, {
      get visible() {
        return w.shared.visible;
      },
      variant: "solid",
      get border_mode() {
        return o(a);
      },
      padding: !1,
      get elem_id() {
        return w.shared.elem_id;
      },
      get elem_classes() {
        return w.shared.elem_classes;
      },
      allow_overflow: !1,
      get container() {
        return w.shared.container;
      },
      get scale() {
        return w.shared.scale;
      },
      get min_width() {
        return w.shared.min_width;
      },
      children: (l, p) => {
        var g = Lu(), M = Ne(g);
        Su(M, Eo(
          {
            get autoscroll() {
              return w.shared.autoscroll;
            },
            get i18n() {
              return w.i18n;
            }
          },
          () => w.shared.loading_status,
          {
            on_clear_status: () => w.dispatch("clear_status", w.shared.loading_status)
          }
        ));
        var A = z(M, 2), L = te(A);
        {
          var G = ($e) => {
            var wr = Iu(), bt = z(te(wr), 2);
            wn(bt, 21, Ae, xn, (en, er) => {
              var Dt = Nu(), tn = te(Dt), Sr = {};
              oe(
                (rn, Ar) => {
                  ge(tn, rn), Sr !== (Sr = Ar) && (Dt.value = (Dt.__value = Ar) ?? "");
                },
                [
                  () => o(er).label || "Label " + String(o(er).group_id),
                  () => ce(o(er).group_id)
                ]
              ), j(en, Dt);
            });
            var Tr;
            ea(bt), oe(() => {
              Tr !== (Tr = o(I)) && (bt.value = (bt.__value = o(I)) ?? "", Cr(bt, o(I)));
            }), Oe("change", bt, Na), j($e, wr);
          }, F = Xe(() => q());
          se(L, ($e) => {
            o(F) && $e(G);
          });
        }
        var K = z(L, 2), ee = te(K), re = z(ee, 2), Ve = z(re, 2), Ke = z(Ve, 2), at = z(Ke, 2);
        let ei;
        var Kr = z(K, 2), ti = te(Kr), ri = te(ti), ja = te(ri), ni = z(ri, 2), Va = te(ni), za = z(ni, 2), Xa = te(za), Er = z(ti, 2);
        let ii;
        var $r = te(Er);
        wt($r, "width", c), wt($r, "height", c), sr($r, ($e) => E = $e, () => E);
        var st = z(Er, 2);
        sr(st, ($e) => m = $e, () => m), sr(Kr, ($e) => v = $e, () => v);
        var Wa = z(Kr, 4), qa = te(Wa);
        oe(
          ($e, wr, bt, Tr, en, er, Dt, tn, Sr, rn, Ar, Za, Ya) => {
            qe(A, $e), ee.disabled = wr, re.disabled = bt, Ve.disabled = Tr, Ke.disabled = en, at.disabled = er, ei = ht(at, 1, "svelte-r41nsf", null, ei, { toggled: o(dt) }), ge(ja, `tx ${Dt ?? ""} · ty ${tn ?? ""}`), ge(Va, `scale ${Sr ?? ""} · rot ${rn ?? ""}°`), ge(Xa, `gain ${Ar ?? ""} · ${o(Ze) ? "focused" : "blur"}`), ii = ht(Er, 1, "loupe svelte-r41nsf", null, ii, { active: o(he) }), qe(Er, Za), qe(st, "cursor:" + o(ne)), ge(qa, Ya);
          },
          [
            () => "min-height:" + de(w.props.height),
            () => !Se(),
            () => !Se(),
            () => !Se(),
            () => !Se(),
            () => !Se(),
            () => o(P).center_x.toFixed(1),
            () => o(P).center_y.toFixed(1),
            () => o(P).scale.toFixed(3),
            () => o(P).rotation_deg.toFixed(1),
            () => o(Ge).toFixed(2),
            () => "left:" + String(Je.x) + "px;top:" + String(Je.y) + "px",
            () => Kt()
          ]
        ), Oe("click", ee, Da), Oe("click", re, Ga), Oe("click", Ve, Fa), Oe("click", Ke, Ua), Oe("click", at, () => {
          x(dt, !o(dt)), De();
        }), Oe("pointerdown", st, Ia), Oe("pointermove", st, Ca), Oe("pointerup", st, $n), Oe("pointercancel", st, $n), Oe("pointerleave", st, Ra), Oe("wheel", st, ka), Oe("focus", st, () => {
          x(Ze, !0);
        }), Oe("blur", st, () => {
          x(Ze, !1);
        }), j(l, g);
      },
      $$slots: { default: !0 }
    });
  }
  Ur();
}
export {
  Cu as default
};

import { i as _n, g as Ci, o as Is, n as Rt, u as fe, s as Ns, r as on, q as Li, c as Ri, a as St, b as Nr, d as wr, e as Ps, T as Bs, f as Cs, h as Ar, j as Tt, p as yn, k as wt, l as Gi, m as Di, t as Ui, v as Gt, w as Fi, x as Dt, y as xn, E as ki, z as a, A as oi, B as Ls, C as ji, D as Vi, F as En, G as Rs, H as ui, I as Gs, J as Ds, K as Ct, L as qe, M as Xi, N as qr, O as Us, P as Fs, Q as ks, R as js, S as Wi, U as Vs, V as zi, W as Sn, X as li, Y as fi, Z as Xs, _ as Ws, $ as zs, a0 as Ys, a1 as qs, a2 as Pr, a3 as Zs, a4 as Js, a5 as $s, a6 as Ks, a7 as Qs, a8 as Tn, a9 as eo, aa as to, ab as ro, ac as no, ad as io, ae as ao, af as Yi, ag as so, ah as oo, ai as Be, aj as un, ak as uo, al as lo, am as fo, an as dt, ao as m, ap as co, aq as ho, ar as po, as as vo, at as go, au as bo, av as q, aw as mo, ax as ci, ay as _o, az as Zr, aA as qi, aB as Zi, aC as z, aD as j, aE as et, aF as yo, aG as hi, aH as $, aI as xt, aJ as _e } from "./render-DlPxDIpa.js";
function Ji(e) {
  throw new Error("https://svelte.dev/e/lifecycle_outside_component");
}
const xo = [];
function Eo(e, t = !1, r = !1) {
  return Er(e, /* @__PURE__ */ new Map(), "", xo, null, r);
}
function Er(e, t, r, n, s = null, u = !1) {
  if (typeof e == "object" && e !== null) {
    var l = t.get(e);
    if (l !== void 0) return l;
    if (e instanceof Map) return (
      /** @type {Snapshot<T>} */
      new Map(e)
    );
    if (e instanceof Set) return (
      /** @type {Snapshot<T>} */
      new Set(e)
    );
    if (_n(e)) {
      var c = (
        /** @type {Snapshot<any>} */
        Array(e.length)
      );
      t.set(e, c), s !== null && t.set(s, c);
      for (var d = 0; d < e.length; d += 1) {
        var h = e[d];
        d in e && (c[d] = Er(h, t, r, n, null, u));
      }
      return c;
    }
    if (Ci(e) === Is) {
      c = {}, t.set(e, c), s !== null && t.set(s, c);
      for (var p of Object.keys(e))
        c[p] = Er(
          // @ts-expect-error
          e[p],
          t,
          r,
          n,
          null,
          u
        );
      return c;
    }
    if (e instanceof Date)
      return (
        /** @type {Snapshot<T>} */
        structuredClone(e)
      );
    if (typeof /** @type {T & { toJSON?: any } } */
    e.toJSON == "function" && !u)
      return Er(
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
function So(e, t, r) {
  if (e == null)
    return t(void 0), r && r(void 0), Rt;
  const n = fe(
    () => e.subscribe(
      t,
      // @ts-expect-error
      r
    )
  );
  return n.unsubscribe ? () => n.unsubscribe() : n;
}
const Bt = [];
function To(e, t) {
  return {
    subscribe: Br(e, t).subscribe
  };
}
function Br(e, t = Rt) {
  let r = null;
  const n = /* @__PURE__ */ new Set();
  function s(c) {
    if (Ns(e, c) && (e = c, r)) {
      const d = !Bt.length;
      for (const h of n)
        h[1](), Bt.push(h, e);
      if (d) {
        for (let h = 0; h < Bt.length; h += 2)
          Bt[h][0](Bt[h + 1]);
        Bt.length = 0;
      }
    }
  }
  function u(c) {
    s(c(
      /** @type {T} */
      e
    ));
  }
  function l(c, d = Rt) {
    const h = [c, d];
    return n.add(h), n.size === 1 && (r = t(s, u) || Rt), c(
      /** @type {T} */
      e
    ), () => {
      n.delete(h), n.size === 0 && r && (r(), r = null);
    };
  }
  return { set: s, update: u, subscribe: l };
}
function jt(e, t, r) {
  const n = !Array.isArray(e), s = n ? [e] : e;
  if (!s.every(Boolean))
    throw new Error("derived() expects stores as input, got a falsy value");
  const u = t.length < 2;
  return To(r, (l, c) => {
    let d = !1;
    const h = [];
    let p = 0, E = Rt;
    const y = () => {
      if (p)
        return;
      E();
      const I = t(n ? h[0] : h, l, c);
      u ? l(I) : E = typeof I == "function" ? I : Rt;
    }, C = s.map(
      (I, B) => So(
        I,
        (O) => {
          h[B] = O, p &= ~(1 << B), d && y();
        },
        () => {
          p |= 1 << B;
        }
      )
    );
    return d = !0, y(), function() {
      on(C), E(), d = !1;
    };
  });
}
let mr = !1;
function wo(e) {
  var t = mr;
  try {
    return mr = !1, [e(), mr];
  } finally {
    mr = t;
  }
}
function Ao(e, t) {
  if (t) {
    const r = document.body;
    e.autofocus = !0, Li(() => {
      document.activeElement === r && e.focus();
    });
  }
}
const Mo = (
  // We gotta write it like this because after downleveling the pure comment may end up in the wrong location
  globalThis?.window?.trustedTypes && /* @__PURE__ */ globalThis.window.trustedTypes.createPolicy("svelte-trusted-html", {
    /** @param {string} html */
    createHTML: (e) => e
  })
);
function Oo(e) {
  return (
    /** @type {string} */
    Mo?.createHTML(e) ?? e
  );
}
function $i(e) {
  var t = Ri("template");
  return t.innerHTML = Oo(e.replaceAll("<!>", "<!---->")), t.content;
}
function ir(e, t) {
  var r = (
    /** @type {Effect} */
    Nr
  );
  r.nodes === null && (r.nodes = { start: e, end: t, a: null, t: null });
}
// @__NO_SIDE_EFFECTS__
function ye(e, t) {
  var r = (t & Bs) !== 0, n = (t & Cs) !== 0, s, u = !e.startsWith("<!>");
  return () => {
    s === void 0 && (s = $i(u ? e : "<!>" + e), r || (s = /** @type {TemplateNode} */
    wr(s)));
    var l = (
      /** @type {TemplateNode} */
      n || Ps ? document.importNode(s, !0) : s.cloneNode(!0)
    );
    if (r) {
      var c = (
        /** @type {TemplateNode} */
        wr(l)
      ), d = (
        /** @type {TemplateNode} */
        l.lastChild
      );
      ir(c, d);
    } else
      ir(l, l);
    return l;
  };
}
// @__NO_SIDE_EFFECTS__
function Ho(e, t, r = "svg") {
  var n = !e.startsWith("<!>"), s = `<${r}>${n ? e : "<!>" + e}</${r}>`, u;
  return () => {
    if (!u) {
      var l = (
        /** @type {DocumentFragment} */
        $i(s)
      ), c = (
        /** @type {Element} */
        wr(l)
      );
      u = /** @type {Element} */
      wr(c);
    }
    var d = (
      /** @type {TemplateNode} */
      u.cloneNode(!0)
    );
    return ir(d, d), d;
  };
}
// @__NO_SIDE_EFFECTS__
function Io(e, t) {
  return /* @__PURE__ */ Ho(e, t, "svg");
}
function No() {
  var e = document.createDocumentFragment(), t = document.createComment(""), r = St();
  return e.append(t, r), ir(t, r), e;
}
function le(e, t) {
  e !== null && e.before(
    /** @type {Node} */
    t
  );
}
class Ki {
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
        Ar(n), this.#n.delete(r);
      else {
        var s = this.#e.get(r);
        s && (Ar(s.effect), this.#r.set(r, s.effect), this.#e.delete(r), s.fragment.lastChild.remove(), this.anchor.before(s.fragment), n = s.effect);
      }
      for (const [u, l] of this.#t) {
        if (this.#t.delete(u), u === t)
          break;
        const c = this.#e.get(l);
        c && (Tt(c.effect), this.#e.delete(l));
      }
      for (const [u, l] of this.#r) {
        if (u === r || this.#n.has(u)) continue;
        const c = () => {
          if (Array.from(this.#t.values()).includes(u)) {
            var h = document.createDocumentFragment();
            Di(l, h), h.append(St()), this.#e.set(u, { effect: l, fragment: h });
          } else
            Tt(l);
          this.#n.delete(u), this.#r.delete(u);
        };
        this.#i || !n ? (this.#n.add(u), yn(l, c, !1)) : c();
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
      r.includes(n) || (Tt(s.effect), this.#e.delete(n));
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
    ), s = Ui();
    if (r && !this.#r.has(t) && !this.#e.has(t))
      if (s) {
        var u = document.createDocumentFragment(), l = St();
        u.append(l), this.#e.set(t, {
          effect: wt(() => r(l)),
          fragment: u
        });
      } else
        this.#r.set(
          t,
          wt(() => r(this.anchor))
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
function Po(e) {
  Gt === null && Ji(), Fi && Gt.l !== null ? Co(Gt).m.push(e) : Dt(() => {
    const t = fe(e);
    if (typeof t == "function") return (
      /** @type {() => void} */
      t
    );
  });
}
function Bo(e) {
  Gt === null && Ji(), Po(() => () => fe(e));
}
function Co(e) {
  var t = (
    /** @type {ComponentContextLegacy} */
    e.l
  );
  return t.u ??= { a: [], b: [], m: [] };
}
function Ye(e, t, r = !1) {
  var n = new Ki(e), s = r ? ki : 0;
  function u(l, c) {
    n.ensure(l, c);
  }
  xn(() => {
    var l = !1;
    t((c, d = 0) => {
      l = !0, u(d, c);
    }), l || u(-1, null);
  }, s);
}
function _r(e, t) {
  return t;
}
function Lo(e, t, r) {
  for (var n = [], s = t.length, u, l = t.length, c = 0; c < s; c++) {
    let E = t[c];
    yn(
      E,
      () => {
        if (u) {
          if (u.pending.delete(E), u.done.add(E), u.pending.size === 0) {
            var y = (
              /** @type {Set<EachOutroGroup>} */
              e.outrogroups
            );
            ln(e, En(u.done)), y.delete(u), y.size === 0 && (e.outrogroups = null);
          }
        } else
          l -= 1;
      },
      !1
    );
  }
  if (l === 0) {
    var d = n.length === 0 && r !== null;
    if (d) {
      var h = (
        /** @type {Element} */
        r
      ), p = (
        /** @type {Element} */
        h.parentNode
      );
      ks(p), p.append(h), e.items.clear();
    }
    ln(e, t, !d);
  } else
    u = {
      pending: new Set(t),
      done: /* @__PURE__ */ new Set()
    }, (e.outrogroups ??= /* @__PURE__ */ new Set()).add(u);
}
function ln(e, t, r = !0) {
  var n;
  if (e.pending.size > 0) {
    n = /* @__PURE__ */ new Set();
    for (const l of e.pending.values())
      for (const c of l)
        n.add(
          /** @type {EachItem} */
          e.items.get(c).e
        );
  }
  for (var s = 0; s < t.length; s++) {
    var u = t[s];
    if (n?.has(u)) {
      u.f |= qe;
      const l = document.createDocumentFragment();
      Di(u, l);
    } else
      Tt(t[s], r);
  }
}
var di;
function yr(e, t, r, n, s, u = null) {
  var l = e, c = /* @__PURE__ */ new Map(), d = (t & ji) !== 0;
  if (d) {
    var h = (
      /** @type {Element} */
      e
    );
    l = h.appendChild(St());
  }
  var p = null, E = Vi(() => {
    var H = r();
    return (
      /** @type {V[]} */
      _n(H) ? H : H == null ? [] : En(H)
    );
  }), y, C = /* @__PURE__ */ new Map(), I = !0;
  function B(H) {
    (M.effect.f & Xi) === 0 && (M.pending.delete(H), M.fallback = p, Ro(M, y, l, t, n), p !== null && (y.length === 0 ? (p.f & qe) === 0 ? Ar(p) : (p.f ^= qe, tr(p, null, l)) : yn(p, () => {
      p = null;
    })));
  }
  function O(H) {
    M.pending.delete(H);
  }
  var S = xn(() => {
    y = /** @type {V[]} */
    a(E);
    for (var H = y.length, _ = /* @__PURE__ */ new Set(), b = (
      /** @type {Batch} */
      Gi
    ), A = Ui(), T = 0; T < H; T += 1) {
      var F = y[T], ee = n(F, T), G = I ? null : c.get(ee);
      G ? (G.v && oi(G.v, F), G.i && oi(G.i, T), A && b.unskip_effect(G.e)) : (G = Go(
        c,
        I ? l : di ??= St(),
        F,
        ee,
        T,
        s,
        t,
        r
      ), I || (G.e.f |= qe), c.set(ee, G)), _.add(ee);
    }
    if (H === 0 && u && !p && (I ? p = wt(() => u(l)) : (p = wt(() => u(di ??= St())), p.f |= qe)), H > _.size && Ls(), !I)
      if (C.set(b, _), A) {
        for (const [vt, Ce] of c)
          _.has(vt) || b.skip_effect(Ce.e);
        b.oncommit(B), b.ondiscard(O);
      } else
        B(b);
    a(E);
  }), M = { effect: S, items: c, pending: C, outrogroups: null, fallback: p };
  I = !1;
}
function Qt(e) {
  for (; e !== null && (e.f & Us) === 0; )
    e = e.next;
  return e;
}
function Ro(e, t, r, n, s) {
  var u = (n & Fs) !== 0, l = t.length, c = e.items, d = Qt(e.effect.first), h, p = null, E, y = [], C = [], I, B, O, S;
  if (u)
    for (S = 0; S < l; S += 1)
      I = t[S], B = s(I, S), O = /** @type {EachItem} */
      c.get(B).e, (O.f & qe) === 0 && (O.nodes?.a?.measure(), (E ??= /* @__PURE__ */ new Set()).add(O));
  for (S = 0; S < l; S += 1) {
    if (I = t[S], B = s(I, S), O = /** @type {EachItem} */
    c.get(B).e, e.outrogroups !== null)
      for (const G of e.outrogroups)
        G.pending.delete(O), G.done.delete(O);
    if ((O.f & qr) !== 0 && (Ar(O), u && (O.nodes?.a?.unfix(), (E ??= /* @__PURE__ */ new Set()).delete(O))), (O.f & qe) !== 0)
      if (O.f ^= qe, O === d)
        tr(O, null, r);
      else {
        var M = p ? p.next : d;
        O === e.effect.last && (e.effect.last = O.prev), O.prev && (O.prev.next = O.next), O.next && (O.next.prev = O.prev), ht(e, p, O), ht(e, O, M), tr(O, M, r), p = O, y = [], C = [], d = Qt(p.next);
        continue;
      }
    if (O !== d) {
      if (h !== void 0 && h.has(O)) {
        if (y.length < C.length) {
          var H = C[0], _;
          p = H.prev;
          var b = y[0], A = y[y.length - 1];
          for (_ = 0; _ < y.length; _ += 1)
            tr(y[_], H, r);
          for (_ = 0; _ < C.length; _ += 1)
            h.delete(C[_]);
          ht(e, b.prev, A.next), ht(e, p, b), ht(e, A, H), d = H, p = A, S -= 1, y = [], C = [];
        } else
          h.delete(O), tr(O, d, r), ht(e, O.prev, O.next), ht(e, O, p === null ? e.effect.first : p.next), ht(e, p, O), p = O;
        continue;
      }
      for (y = [], C = []; d !== null && d !== O; )
        (h ??= /* @__PURE__ */ new Set()).add(d), C.push(d), d = Qt(d.next);
      if (d === null)
        continue;
    }
    (O.f & qe) === 0 && y.push(O), p = O, d = Qt(O.next);
  }
  if (e.outrogroups !== null) {
    for (const G of e.outrogroups)
      G.pending.size === 0 && (ln(e, En(G.done)), e.outrogroups?.delete(G));
    e.outrogroups.size === 0 && (e.outrogroups = null);
  }
  if (d !== null || h !== void 0) {
    var T = [];
    if (h !== void 0)
      for (O of h)
        (O.f & qr) === 0 && T.push(O);
    for (; d !== null; )
      (d.f & qr) === 0 && d !== e.fallback && T.push(d), d = Qt(d.next);
    var F = T.length;
    if (F > 0) {
      var ee = (n & ji) !== 0 && l === 0 ? r : null;
      if (u) {
        for (S = 0; S < F; S += 1)
          T[S].nodes?.a?.measure();
        for (S = 0; S < F; S += 1)
          T[S].nodes?.a?.fix();
      }
      Lo(e, T, ee);
    }
  }
  u && Li(() => {
    if (E !== void 0)
      for (O of E)
        O.nodes?.a?.apply();
  });
}
function Go(e, t, r, n, s, u, l, c) {
  var d = (l & Gs) !== 0 ? (l & Ds) === 0 ? Ct(r, !1, !1) : ui(r) : null, h = (l & Rs) !== 0 ? ui(s) : null;
  return {
    v: d,
    i: h,
    e: wt(() => (u(t, d ?? r, h ?? s, c), () => {
      e.delete(n);
    }))
  };
}
function tr(e, t, r) {
  if (e.nodes)
    for (var n = e.nodes.start, s = e.nodes.end, u = t && (t.f & qe) === 0 ? (
      /** @type {EffectNodes} */
      t.nodes.start
    ) : r; n !== null; ) {
      var l = (
        /** @type {TemplateNode} */
        js(n)
      );
      if (u.before(n), n === s)
        return;
      n = l;
    }
}
function ht(e, t, r) {
  t === null ? e.effect.first = r : t.next = r, r === null ? e.effect.last = t : r.prev = t;
}
function Do(e, t, r, n, s) {
  var u = t.$$slots?.[r], l = !1;
  u === !0 && (u = t.children, l = !0), u === void 0 || u(e, l ? () => n : n);
}
function Uo(e, t, r, n, s, u) {
  var l = null, c = (
    /** @type {TemplateNode} */
    e
  ), d = new Ki(c, !1);
  xn(() => {
    const h = t() || null;
    var p = h === "svg" ? Vs : void 0;
    if (h === null) {
      d.ensure(null, null);
      return;
    }
    return d.ensure(h, (E) => {
      if (h) {
        if (l = Ri(h, p), ir(l, l), n) {
          var y = null, C = l.appendChild(St());
          n(l, C), y?.remove();
        }
        Nr.nodes.end = l, E.before(l);
      }
    }), () => {
    };
  }, ki), Wi(() => {
  });
}
function Fo(e, t) {
  var r = void 0, n;
  zi(() => {
    r !== (r = t()) && (n && (Tt(n), n = null), r && (n = wt(() => {
      Sn(() => (
        /** @type {(node: Element) => void} */
        r(e)
      ));
    })));
  });
}
function Qi(e) {
  var t, r, n = "";
  if (typeof e == "string" || typeof e == "number") n += e;
  else if (typeof e == "object") if (Array.isArray(e)) {
    var s = e.length;
    for (t = 0; t < s; t++) e[t] && (r = Qi(e[t])) && (n && (n += " "), n += r);
  } else for (r in e) e[r] && (n && (n += " "), n += r);
  return n;
}
function ko() {
  for (var e, t, r = 0, n = "", s = arguments.length; r < s; r++) (e = arguments[r]) && (t = Qi(e)) && (n && (n += " "), n += t);
  return n;
}
function jo(e) {
  return typeof e == "object" ? ko(e) : e ?? "";
}
const pi = [...` 	
\r\f \v\uFEFF`];
function Vo(e, t, r) {
  var n = e == null ? "" : "" + e;
  if (t && (n = n ? n + " " + t : t), r) {
    for (var s of Object.keys(r))
      if (r[s])
        n = n ? n + " " + s : s;
      else if (n.length)
        for (var u = s.length, l = 0; (l = n.indexOf(s, l)) >= 0; ) {
          var c = l + u;
          (l === 0 || pi.includes(n[l - 1])) && (c === n.length || pi.includes(n[c])) ? n = (l === 0 ? "" : n.substring(0, l)) + n.substring(c + 1) : l = c;
        }
  }
  return n === "" ? null : n;
}
function vi(e, t = !1) {
  var r = t ? " !important;" : ";", n = "";
  for (var s of Object.keys(e)) {
    var u = e[s];
    u != null && u !== "" && (n += " " + s + ": " + u + r);
  }
  return n;
}
function Jr(e) {
  return e[0] !== "-" || e[1] !== "-" ? e.toLowerCase() : e;
}
function Xo(e, t) {
  if (t) {
    var r = "", n, s;
    if (Array.isArray(t) ? (n = t[0], s = t[1]) : n = t, e) {
      e = String(e).replaceAll(/\s*\/\*.*?\*\/\s*/g, "").trim();
      var u = !1, l = 0, c = !1, d = [];
      n && d.push(...Object.keys(n).map(Jr)), s && d.push(...Object.keys(s).map(Jr));
      var h = 0, p = -1;
      const B = e.length;
      for (var E = 0; E < B; E++) {
        var y = e[E];
        if (c ? y === "/" && e[E - 1] === "*" && (c = !1) : u ? u === y && (u = !1) : y === "/" && e[E + 1] === "*" ? c = !0 : y === '"' || y === "'" ? u = y : y === "(" ? l++ : y === ")" && l--, !c && u === !1 && l === 0) {
          if (y === ":" && p === -1)
            p = E;
          else if (y === ";" || E === B - 1) {
            if (p !== -1) {
              var C = Jr(e.substring(h, p).trim());
              if (!d.includes(C)) {
                y !== ";" && E++;
                var I = e.substring(h, E).trim();
                r += " " + I + ";";
              }
            }
            h = E + 1, p = -1;
          }
        }
      }
    }
    return n && (r += vi(n)), s && (r += vi(s, !0)), r = r.trim(), r === "" ? null : r;
  }
  return e == null ? null : String(e);
}
function Et(e, t, r, n, s, u) {
  var l = (
    /** @type {any} */
    e[li]
  );
  if (l !== r || l === void 0) {
    var c = Vo(r, n, u);
    c == null ? e.removeAttribute("class") : t ? e.className = c : e.setAttribute("class", c), e[li] = r;
  } else if (u && s !== u)
    for (var d in u) {
      var h = !!u[d];
      (s == null || h !== !!s[d]) && e.classList.toggle(d, h);
    }
  return u;
}
function $r(e, t = {}, r, n) {
  for (var s in r) {
    var u = r[s];
    t[s] !== u && (r[s] == null ? e.style.removeProperty(s) : e.style.setProperty(s, u, n));
  }
}
function Mr(e, t, r, n) {
  var s = (
    /** @type {any} */
    e[fi]
  );
  if (s !== t) {
    var u = Xo(t, n);
    u == null ? e.removeAttribute("style") : e.style.cssText = u, e[fi] = t;
  } else n && (Array.isArray(n) ? ($r(e, r?.[0], n[0]), $r(e, r?.[1], n[1], "important")) : $r(e, r, n));
  return n;
}
function Or(e, t, r = !1) {
  if (e.multiple) {
    if (t == null)
      return;
    if (!_n(t))
      return Xs();
    for (var n of e.options)
      n.selected = t.includes(gi(n));
    return;
  }
  for (n of e.options) {
    var s = gi(n);
    if (Ws(s, t)) {
      n.selected = !0;
      return;
    }
  }
  (!r || t !== void 0) && (e.selectedIndex = -1);
}
function ea(e) {
  var t = new MutationObserver(() => {
    Or(e, e.__value);
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
  }), Wi(() => {
    t.disconnect();
  });
}
function gi(e) {
  return "__value" in e ? e.__value : e.value;
}
const rr = /* @__PURE__ */ Symbol("class"), Lt = /* @__PURE__ */ Symbol("style"), ta = /* @__PURE__ */ Symbol("is custom element"), ra = /* @__PURE__ */ Symbol("is html"), Wo = Pr ? "input" : "INPUT", zo = Pr ? "option" : "OPTION", Yo = Pr ? "select" : "SELECT", qo = Pr ? "progress" : "PROGRESS";
function Zo(e, t) {
  var r = wn(e);
  r.value === (r.value = // treat null and undefined the same for the initial value
  t ?? void 0) || // @ts-expect-error
  // `progress` elements always need their value set when it's `0`
  e.value === t && (t !== 0 || e.nodeName !== qo) || (e.value = t ?? "");
}
function Jo(e, t) {
  t ? e.hasAttribute("selected") || e.setAttribute("selected", "") : e.removeAttribute("selected");
}
function nr(e, t, r, n) {
  var s = wn(e);
  s[t] !== (s[t] = r) && (t === "loading" && (e[zs] = r), r == null ? e.removeAttribute(t) : typeof r != "string" && na(e).includes(t) ? e[t] = r : e.setAttribute(t, r));
}
function $o(e, t, r, n, s = !1, u = !1) {
  var l = wn(e), c = l[ta], d = !l[ra], h = t || {}, p = e.nodeName === zo;
  for (var E in t)
    E in r || (r[E] = null);
  r.class ? r.class = jo(r.class) : r.class = null, r[Lt] && (r.style ??= null);
  var y = na(e);
  if (e.nodeName === Wo && "type" in r && ("value" in r || "__value" in r)) {
    var C = r.type;
    (C !== h.type || C === void 0 && e.hasAttribute("type")) && (h.type = C, nr(e, "type", C));
  }
  for (const _ in r) {
    let b = r[_];
    if (p && _ === "value" && b == null) {
      e.value = e.__value = "", h[_] = b;
      continue;
    }
    if (_ === "class") {
      var I = e.namespaceURI === "http://www.w3.org/1999/xhtml";
      Et(e, I, b, n, t?.[rr], r[rr]), h[_] = b, h[rr] = r[rr];
      continue;
    }
    if (_ === "style") {
      Mr(e, b, t?.[Lt], r[Lt]), h[_] = b, h[Lt] = r[Lt];
      continue;
    }
    var B = h[_];
    if (!(b === B && !(b === void 0 && e.hasAttribute(_)))) {
      h[_] = b;
      var O = _[0] + _[1];
      if (O !== "$$")
        if (O === "on") {
          const A = {}, T = "$$" + _;
          let F = _.slice(2);
          var S = no(F);
          if (Ks(F) && (F = F.slice(0, -7), A.capture = !0), !S && B) {
            if (b != null) continue;
            e.removeEventListener(F, h[T], A), h[T] = null;
          }
          if (S)
            Qs(F, e, b), Tn([F]);
          else if (b != null) {
            let ee = function(G) {
              h[_].call(this, G);
            };
            h[T] = eo(F, e, ee, A);
          }
        } else if (_ === "style")
          nr(e, _, b);
        else if (_ === "autofocus")
          Ao(
            /** @type {HTMLElement} */
            e,
            !!b
          );
        else if (!c && (_ === "__value" || _ === "value" && b != null))
          e.value = e.__value = b;
        else if (_ === "selected" && p)
          Jo(
            /** @type {HTMLOptionElement} */
            e,
            b
          );
        else {
          var M = _;
          d || (M = to(M));
          var H = M === "defaultValue" || M === "defaultChecked";
          if (b == null && !c && !H)
            if (l[_] = null, M === "value" || M === "checked") {
              let A = (
                /** @type {HTMLInputElement} */
                e
              );
              const T = t === void 0;
              if (M === "value") {
                let F = A.defaultValue;
                A.removeAttribute(M), A.defaultValue = F, A.value = A.__value = T ? F : null;
              } else {
                let F = A.defaultChecked;
                A.removeAttribute(M), A.defaultChecked = F, A.checked = T ? F : !1;
              }
            } else
              e.removeAttribute(_);
          else H || y.includes(M) && (c || typeof b != "string") ? (e[M] = b, M in l && (l[M] = ro)) : typeof b != "function" && nr(e, M, b);
        }
    }
  }
  return h;
}
function Ko(e, t, r = [], n = [], s = [], u, l = !1, c = !1) {
  Js(s, r, n, (d) => {
    var h = void 0, p = {}, E = e.nodeName === Yo, y = !1;
    if (zi(() => {
      var I = t(...d.map(a)), B = $o(
        e,
        h,
        I,
        u,
        l,
        c
      );
      y && E && "value" in I && Or(
        /** @type {HTMLSelectElement} */
        e,
        I.value
      );
      for (let S of Object.getOwnPropertySymbols(p))
        I[S] || Tt(p[S]);
      for (let S of Object.getOwnPropertySymbols(I)) {
        var O = I[S];
        S.description === $s && (!h || O !== h[S]) && (p[S] && Tt(p[S]), p[S] = wt(() => Fo(e, () => O))), B[S] = O;
      }
      h = B;
    }), E) {
      var C = (
        /** @type {HTMLSelectElement} */
        e
      );
      Sn(() => {
        Or(
          C,
          /** @type {Record<string | symbol, any>} */
          h.value,
          !0
        ), ea(C);
      });
    }
    y = !0;
  });
}
function wn(e) {
  return (
    /** @type {Record<string | symbol, unknown>} **/
    /** @type {any} */
    e[Ys] ??= {
      [ta]: e.nodeName.includes("-"),
      [ra]: e.namespaceURI === qs
    }
  );
}
var bi = /* @__PURE__ */ new Map();
function na(e) {
  var t = e.getAttribute("is") || e.nodeName, r = bi.get(t);
  if (r) return r;
  bi.set(t, r = []);
  for (var n, s = e, u = Element.prototype; u !== s; ) {
    n = Zs(s);
    for (var l in n)
      n[l].set && // better safe than sorry, we don't want spread attributes to mess with HTML content
      l !== "innerHTML" && l !== "textContent" && l !== "innerText" && r.push(l);
    s = Ci(s);
  }
  return r;
}
function Kr(e, t) {
  return e === t || e?.[Yi] === t;
}
function ia(e = {}, t, r, n) {
  var s = (
    /** @type {ComponentContext} */
    Gt.r
  ), u = (
    /** @type {Effect} */
    Nr
  );
  return Sn(() => {
    var l, c;
    return io(() => {
      l = c, c = [], fe(() => {
        Kr(r(...c), e) || (t(e, ...c), l && Kr(r(...l), e) && t(null, ...l));
      });
    }), () => {
      let d = u;
      for (; d !== s && d.parent !== null && d.parent.f & ao; )
        d = d.parent;
      const h = () => {
        c && Kr(r(...c), e) && t(null, ...c);
      }, p = d.teardown;
      d.teardown = () => {
        h(), p?.();
      };
    };
  }), e;
}
function Qo(e = !1) {
  const t = (
    /** @type {ComponentContextLegacy} */
    Gt
  ), r = t.l.u;
  if (!r) return;
  let n = () => Be(t.s);
  if (e) {
    let s = 0, u = (
      /** @type {Record<string, any>} */
      {}
    );
    const l = un(() => {
      let c = !1;
      const d = t.s;
      for (const h in d)
        d[h] !== u[h] && (u[h] = d[h], c = !0);
      return c && s++, s;
    });
    n = () => a(l);
  }
  r.b.length && so(() => {
    mi(t, n), on(r.b);
  }), Dt(() => {
    const s = fe(() => r.m.map(oo));
    return () => {
      for (const u of s)
        typeof u == "function" && u();
    };
  }), r.a.length && Dt(() => {
    mi(t, n), on(r.a);
  });
}
function mi(e, t) {
  if (e.l.s)
    for (const r of e.l.s) a(r);
  t();
}
const eu = {
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
function tu(e, t, r) {
  return new Proxy(
    { props: e, exclude: t },
    eu
  );
}
function Q(e, t, r, n) {
  var s = !Fi || (r & ho) !== 0, u = (r & co) !== 0, l = (r & vo) !== 0, c = (
    /** @type {V} */
    n
  ), d = !0, h = (
    /** @type {Derived<V> | undefined} */
    void 0
  ), p = () => l && s ? (h ??= un(
    /** @type {() => V} */
    n
  ), a(h)) : (d && (d = !1, c = l ? fe(
    /** @type {() => V} */
    n
  ) : (
    /** @type {V} */
    n
  )), c);
  let E;
  if (u) {
    var y = Yi in e || bo in e;
    E = uo(e, t)?.set ?? (y && t in e ? (_) => e[t] = _ : void 0);
  }
  var C, I = !1;
  u ? [C, I] = wo(() => (
    /** @type {V} */
    e[t]
  )) : C = /** @type {V} */
  e[t], C === void 0 && n !== void 0 && (C = p(), E && (s && lo(), E(C)));
  var B;
  if (s ? B = () => {
    var _ = (
      /** @type {V} */
      e[t]
    );
    return _ === void 0 ? p() : (d = !0, _);
  } : B = () => {
    var _ = (
      /** @type {V} */
      e[t]
    );
    return _ !== void 0 && (c = /** @type {V} */
    void 0), _ === void 0 ? c : _;
  }, s && (r & fo) === 0)
    return B;
  if (E) {
    var O = e.$$legacy;
    return (
      /** @type {() => V} */
      (function(_, b) {
        return arguments.length > 0 ? ((!s || !b || O || I) && E(b ? B() : _), _) : B();
      })
    );
  }
  var S = !1, M = ((r & po) !== 0 ? un : Vi)(() => (S = !1, B()));
  u && a(M);
  var H = (
    /** @type {Effect} */
    Nr
  );
  return (
    /** @type {() => V} */
    (function(_, b) {
      if (arguments.length > 0) {
        const A = b ? a(M) : s && u ? dt(_) : _;
        return m(M, A), S = !0, c !== void 0 && (c = A), _;
      }
      return go && S || (H.f & Xi) !== 0 ? M.v : a(M);
    })
  );
}
const ru = [
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
], _i = {
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
ru.reduce((e, { color: t, primary: r, secondary: n }) => ({
  ...e,
  [t]: {
    primary: _i[t][r],
    secondary: _i[t][n]
  }
}), {});
function nu(e) {
  return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, "default") ? e.default : e;
}
var Qr, yi;
function iu() {
  if (yi) return Qr;
  yi = 1;
  var e = function(M) {
    return t(M) && !r(M);
  };
  function t(S) {
    return !!S && typeof S == "object";
  }
  function r(S) {
    var M = Object.prototype.toString.call(S);
    return M === "[object RegExp]" || M === "[object Date]" || u(S);
  }
  var n = typeof Symbol == "function" && Symbol.for, s = n ? /* @__PURE__ */ Symbol.for("react.element") : 60103;
  function u(S) {
    return S.$$typeof === s;
  }
  function l(S) {
    return Array.isArray(S) ? [] : {};
  }
  function c(S, M) {
    return M.clone !== !1 && M.isMergeableObject(S) ? B(l(S), S, M) : S;
  }
  function d(S, M, H) {
    return S.concat(M).map(function(_) {
      return c(_, H);
    });
  }
  function h(S, M) {
    if (!M.customMerge)
      return B;
    var H = M.customMerge(S);
    return typeof H == "function" ? H : B;
  }
  function p(S) {
    return Object.getOwnPropertySymbols ? Object.getOwnPropertySymbols(S).filter(function(M) {
      return Object.propertyIsEnumerable.call(S, M);
    }) : [];
  }
  function E(S) {
    return Object.keys(S).concat(p(S));
  }
  function y(S, M) {
    try {
      return M in S;
    } catch {
      return !1;
    }
  }
  function C(S, M) {
    return y(S, M) && !(Object.hasOwnProperty.call(S, M) && Object.propertyIsEnumerable.call(S, M));
  }
  function I(S, M, H) {
    var _ = {};
    return H.isMergeableObject(S) && E(S).forEach(function(b) {
      _[b] = c(S[b], H);
    }), E(M).forEach(function(b) {
      C(S, b) || (y(S, b) && H.isMergeableObject(M[b]) ? _[b] = h(b, H)(S[b], M[b], H) : _[b] = c(M[b], H));
    }), _;
  }
  function B(S, M, H) {
    H = H || {}, H.arrayMerge = H.arrayMerge || d, H.isMergeableObject = H.isMergeableObject || e, H.cloneUnlessOtherwiseSpecified = c;
    var _ = Array.isArray(M), b = Array.isArray(S), A = _ === b;
    return A ? _ ? H.arrayMerge(S, M, H) : I(S, M, H) : c(M, H);
  }
  B.all = function(M, H) {
    if (!Array.isArray(M))
      throw new Error("first argument should be an array");
    return M.reduce(function(_, b) {
      return B(_, b, H);
    }, {});
  };
  var O = B;
  return Qr = O, Qr;
}
var au = iu();
const su = /* @__PURE__ */ nu(au);
var fn = function(e, t) {
  return fn = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(r, n) {
    r.__proto__ = n;
  } || function(r, n) {
    for (var s in n) Object.prototype.hasOwnProperty.call(n, s) && (r[s] = n[s]);
  }, fn(e, t);
};
function Cr(e, t) {
  if (typeof t != "function" && t !== null)
    throw new TypeError("Class extends value " + String(t) + " is not a constructor or null");
  fn(e, t);
  function r() {
    this.constructor = e;
  }
  e.prototype = t === null ? Object.create(t) : (r.prototype = t.prototype, new r());
}
var V = function() {
  return V = Object.assign || function(t) {
    for (var r, n = 1, s = arguments.length; n < s; n++) {
      r = arguments[n];
      for (var u in r) Object.prototype.hasOwnProperty.call(r, u) && (t[u] = r[u]);
    }
    return t;
  }, V.apply(this, arguments);
};
function ou(e, t) {
  var r = {};
  for (var n in e) Object.prototype.hasOwnProperty.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && typeof Object.getOwnPropertySymbols == "function")
    for (var s = 0, n = Object.getOwnPropertySymbols(e); s < n.length; s++)
      t.indexOf(n[s]) < 0 && Object.prototype.propertyIsEnumerable.call(e, n[s]) && (r[n[s]] = e[n[s]]);
  return r;
}
function en(e, t, r) {
  if (r || arguments.length === 2) for (var n = 0, s = t.length, u; n < s; n++)
    (u || !(n in t)) && (u || (u = Array.prototype.slice.call(t, 0, n)), u[n] = t[n]);
  return e.concat(u || Array.prototype.slice.call(t));
}
function tn(e, t) {
  var r = t && t.cache ? t.cache : pu, n = t && t.serializer ? t.serializer : hu, s = t && t.strategy ? t.strategy : fu;
  return s(e, {
    cache: r,
    serializer: n
  });
}
function uu(e) {
  return e == null || typeof e == "number" || typeof e == "boolean";
}
function lu(e, t, r, n) {
  var s = uu(n) ? n : r(n), u = t.get(s);
  return typeof u > "u" && (u = e.call(this, n), t.set(s, u)), u;
}
function aa(e, t, r) {
  var n = Array.prototype.slice.call(arguments, 3), s = r(n), u = t.get(s);
  return typeof u > "u" && (u = e.apply(this, n), t.set(s, u)), u;
}
function sa(e, t, r, n, s) {
  return r.bind(t, e, n, s);
}
function fu(e, t) {
  var r = e.length === 1 ? lu : aa;
  return sa(e, this, r, t.cache.create(), t.serializer);
}
function cu(e, t) {
  return sa(e, this, aa, t.cache.create(), t.serializer);
}
var hu = function() {
  return JSON.stringify(arguments);
}, du = (
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
), pu = {
  create: function() {
    return new du();
  }
}, rn = {
  variadic: cu
}, D;
(function(e) {
  e[e.EXPECT_ARGUMENT_CLOSING_BRACE = 1] = "EXPECT_ARGUMENT_CLOSING_BRACE", e[e.EMPTY_ARGUMENT = 2] = "EMPTY_ARGUMENT", e[e.MALFORMED_ARGUMENT = 3] = "MALFORMED_ARGUMENT", e[e.EXPECT_ARGUMENT_TYPE = 4] = "EXPECT_ARGUMENT_TYPE", e[e.INVALID_ARGUMENT_TYPE = 5] = "INVALID_ARGUMENT_TYPE", e[e.EXPECT_ARGUMENT_STYLE = 6] = "EXPECT_ARGUMENT_STYLE", e[e.INVALID_NUMBER_SKELETON = 7] = "INVALID_NUMBER_SKELETON", e[e.INVALID_DATE_TIME_SKELETON = 8] = "INVALID_DATE_TIME_SKELETON", e[e.EXPECT_NUMBER_SKELETON = 9] = "EXPECT_NUMBER_SKELETON", e[e.EXPECT_DATE_TIME_SKELETON = 10] = "EXPECT_DATE_TIME_SKELETON", e[e.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE = 11] = "UNCLOSED_QUOTE_IN_ARGUMENT_STYLE", e[e.EXPECT_SELECT_ARGUMENT_OPTIONS = 12] = "EXPECT_SELECT_ARGUMENT_OPTIONS", e[e.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE = 13] = "EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE = 14] = "INVALID_PLURAL_ARGUMENT_OFFSET_VALUE", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR = 15] = "EXPECT_SELECT_ARGUMENT_SELECTOR", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR = 16] = "EXPECT_PLURAL_ARGUMENT_SELECTOR", e[e.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT = 17] = "EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT", e[e.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT = 18] = "EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT", e[e.INVALID_PLURAL_ARGUMENT_SELECTOR = 19] = "INVALID_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_PLURAL_ARGUMENT_SELECTOR = 20] = "DUPLICATE_PLURAL_ARGUMENT_SELECTOR", e[e.DUPLICATE_SELECT_ARGUMENT_SELECTOR = 21] = "DUPLICATE_SELECT_ARGUMENT_SELECTOR", e[e.MISSING_OTHER_CLAUSE = 22] = "MISSING_OTHER_CLAUSE", e[e.INVALID_TAG = 23] = "INVALID_TAG", e[e.INVALID_TAG_NAME = 25] = "INVALID_TAG_NAME", e[e.UNMATCHED_CLOSING_TAG = 26] = "UNMATCHED_CLOSING_TAG", e[e.UNCLOSED_TAG = 27] = "UNCLOSED_TAG";
})(D || (D = {}));
var K;
(function(e) {
  e[e.literal = 0] = "literal", e[e.argument = 1] = "argument", e[e.number = 2] = "number", e[e.date = 3] = "date", e[e.time = 4] = "time", e[e.select = 5] = "select", e[e.plural = 6] = "plural", e[e.pound = 7] = "pound", e[e.tag = 8] = "tag";
})(K || (K = {}));
var Ut;
(function(e) {
  e[e.number = 0] = "number", e[e.dateTime = 1] = "dateTime";
})(Ut || (Ut = {}));
function xi(e) {
  return e.type === K.literal;
}
function vu(e) {
  return e.type === K.argument;
}
function oa(e) {
  return e.type === K.number;
}
function ua(e) {
  return e.type === K.date;
}
function la(e) {
  return e.type === K.time;
}
function fa(e) {
  return e.type === K.select;
}
function ca(e) {
  return e.type === K.plural;
}
function gu(e) {
  return e.type === K.pound;
}
function ha(e) {
  return e.type === K.tag;
}
function da(e) {
  return !!(e && typeof e == "object" && e.type === Ut.number);
}
function cn(e) {
  return !!(e && typeof e == "object" && e.type === Ut.dateTime);
}
var pa = /[ \xA0\u1680\u2000-\u200A\u202F\u205F\u3000]/, bu = /(?:[Eec]{1,6}|G{1,5}|[Qq]{1,5}|(?:[yYur]+|U{1,5})|[ML]{1,5}|d{1,2}|D{1,3}|F{1}|[abB]{1,5}|[hkHK]{1,2}|w{1,2}|W{1}|m{1,2}|s{1,2}|[zZOvVxX]{1,4})(?=([^']*'[^']*')*[^']*$)/g;
function mu(e) {
  var t = {};
  return e.replace(bu, function(r) {
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
var _u = /[\t-\r \x85\u200E\u200F\u2028\u2029]/i;
function yu(e) {
  if (e.length === 0)
    throw new Error("Number skeleton cannot be empty");
  for (var t = e.split(_u).filter(function(y) {
    return y.length > 0;
  }), r = [], n = 0, s = t; n < s.length; n++) {
    var u = s[n], l = u.split("/");
    if (l.length === 0)
      throw new Error("Invalid number skeleton");
    for (var c = l[0], d = l.slice(1), h = 0, p = d; h < p.length; h++) {
      var E = p[h];
      if (E.length === 0)
        throw new Error("Invalid number skeleton");
    }
    r.push({ stem: c, options: d });
  }
  return r;
}
function xu(e) {
  return e.replace(/^(.*?)-/, "");
}
var Ei = /^\.(?:(0+)(\*)?|(#+)|(0+)(#+))$/g, va = /^(@+)?(\+|#+)?[rs]?$/g, Eu = /(\*)(0+)|(#+)(0+)|(0+)/g, ga = /^(0+)$/;
function Si(e) {
  var t = {};
  return e[e.length - 1] === "r" ? t.roundingPriority = "morePrecision" : e[e.length - 1] === "s" && (t.roundingPriority = "lessPrecision"), e.replace(va, function(r, n, s) {
    return typeof s != "string" ? (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length) : s === "+" ? t.minimumSignificantDigits = n.length : n[0] === "#" ? t.maximumSignificantDigits = n.length : (t.minimumSignificantDigits = n.length, t.maximumSignificantDigits = n.length + (typeof s == "string" ? s.length : 0)), "";
  }), t;
}
function ba(e) {
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
function Su(e) {
  var t;
  if (e[0] === "E" && e[1] === "E" ? (t = {
    notation: "engineering"
  }, e = e.slice(2)) : e[0] === "E" && (t = {
    notation: "scientific"
  }, e = e.slice(1)), t) {
    var r = e.slice(0, 2);
    if (r === "+!" ? (t.signDisplay = "always", e = e.slice(2)) : r === "+?" && (t.signDisplay = "exceptZero", e = e.slice(2)), !ga.test(e))
      throw new Error("Malformed concise eng/scientific notation");
    t.minimumIntegerDigits = e.length;
  }
  return t;
}
function Ti(e) {
  var t = {}, r = ba(e);
  return r || t;
}
function Tu(e) {
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
        t.style = "unit", t.unit = xu(s.options[0]);
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
        t = V(V(V({}, t), { notation: "scientific" }), s.options.reduce(function(d, h) {
          return V(V({}, d), Ti(h));
        }, {}));
        continue;
      case "engineering":
        t = V(V(V({}, t), { notation: "engineering" }), s.options.reduce(function(d, h) {
          return V(V({}, d), Ti(h));
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
        s.options[0].replace(Eu, function(d, h, p, E, y, C) {
          if (h)
            t.minimumIntegerDigits = p.length;
          else {
            if (E && y)
              throw new Error("We currently do not support maximum integer digits");
            if (C)
              throw new Error("We currently do not support exact integer digits");
          }
          return "";
        });
        continue;
    }
    if (ga.test(s.stem)) {
      t.minimumIntegerDigits = s.stem.length;
      continue;
    }
    if (Ei.test(s.stem)) {
      if (s.options.length > 1)
        throw new RangeError("Fraction-precision stems only accept a single optional option");
      s.stem.replace(Ei, function(d, h, p, E, y, C) {
        return p === "*" ? t.minimumFractionDigits = h.length : E && E[0] === "#" ? t.maximumFractionDigits = E.length : y && C ? (t.minimumFractionDigits = y.length, t.maximumFractionDigits = y.length + C.length) : (t.minimumFractionDigits = h.length, t.maximumFractionDigits = h.length), "";
      });
      var u = s.options[0];
      u === "w" ? t = V(V({}, t), { trailingZeroDisplay: "stripIfInteger" }) : u && (t = V(V({}, t), Si(u)));
      continue;
    }
    if (va.test(s.stem)) {
      t = V(V({}, t), Si(s.stem));
      continue;
    }
    var l = ba(s.stem);
    l && (t = V(V({}, t), l));
    var c = Su(s.stem);
    c && (t = V(V({}, t), c));
  }
  return t;
}
var xr = {
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
function wu(e, t) {
  for (var r = "", n = 0; n < e.length; n++) {
    var s = e.charAt(n);
    if (s === "j") {
      for (var u = 0; n + 1 < e.length && e.charAt(n + 1) === s; )
        u++, n++;
      var l = 1 + (u & 1), c = u < 2 ? 1 : 3 + (u >> 1), d = "a", h = Au(t);
      for ((h == "H" || h == "k") && (c = 0); c-- > 0; )
        r += d;
      for (; l-- > 0; )
        r = h + r;
    } else s === "J" ? r += "H" : r += s;
  }
  return r;
}
function Au(e) {
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
  var s = xr[n || ""] || xr[r || ""] || xr["".concat(r, "-001")] || xr["001"];
  return s[0];
}
var nn, Mu = new RegExp("^".concat(pa.source, "*")), Ou = new RegExp("".concat(pa.source, "*$"));
function U(e, t) {
  return { start: e, end: t };
}
var Hu = !!String.prototype.startsWith && "_a".startsWith("a", 1), Iu = !!String.fromCodePoint, Nu = !!Object.fromEntries, Pu = !!String.prototype.codePointAt, Bu = !!String.prototype.trimStart, Cu = !!String.prototype.trimEnd, Lu = !!Number.isSafeInteger, Ru = Lu ? Number.isSafeInteger : function(e) {
  return typeof e == "number" && isFinite(e) && Math.floor(e) === e && Math.abs(e) <= 9007199254740991;
}, hn = !0;
try {
  var Gu = _a("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  hn = ((nn = Gu.exec("a")) === null || nn === void 0 ? void 0 : nn[0]) === "a";
} catch {
  hn = !1;
}
var wi = Hu ? (
  // Native
  function(t, r, n) {
    return t.startsWith(r, n);
  }
) : (
  // For IE11
  function(t, r, n) {
    return t.slice(n, n + r.length) === r;
  }
), dn = Iu ? String.fromCodePoint : (
  // IE11
  function() {
    for (var t = [], r = 0; r < arguments.length; r++)
      t[r] = arguments[r];
    for (var n = "", s = t.length, u = 0, l; s > u; ) {
      if (l = t[u++], l > 1114111)
        throw RangeError(l + " is not a valid code point");
      n += l < 65536 ? String.fromCharCode(l) : String.fromCharCode(((l -= 65536) >> 10) + 55296, l % 1024 + 56320);
    }
    return n;
  }
), Ai = (
  // native
  Nu ? Object.fromEntries : (
    // Ponyfill
    function(t) {
      for (var r = {}, n = 0, s = t; n < s.length; n++) {
        var u = s[n], l = u[0], c = u[1];
        r[l] = c;
      }
      return r;
    }
  )
), ma = Pu ? (
  // Native
  function(t, r) {
    return t.codePointAt(r);
  }
) : (
  // IE 11
  function(t, r) {
    var n = t.length;
    if (!(r < 0 || r >= n)) {
      var s = t.charCodeAt(r), u;
      return s < 55296 || s > 56319 || r + 1 === n || (u = t.charCodeAt(r + 1)) < 56320 || u > 57343 ? s : (s - 55296 << 10) + (u - 56320) + 65536;
    }
  }
), Du = Bu ? (
  // Native
  function(t) {
    return t.trimStart();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(Mu, "");
  }
), Uu = Cu ? (
  // Native
  function(t) {
    return t.trimEnd();
  }
) : (
  // Ponyfill
  function(t) {
    return t.replace(Ou, "");
  }
);
function _a(e, t) {
  return new RegExp(e, t);
}
var pn;
if (hn) {
  var Mi = _a("([^\\p{White_Space}\\p{Pattern_Syntax}]*)", "yu");
  pn = function(t, r) {
    var n;
    Mi.lastIndex = r;
    var s = Mi.exec(t);
    return (n = s[1]) !== null && n !== void 0 ? n : "";
  };
} else
  pn = function(t, r) {
    for (var n = []; ; ) {
      var s = ma(t, r);
      if (s === void 0 || ya(s) || Vu(s))
        break;
      n.push(s), r += s >= 65536 ? 2 : 1;
    }
    return dn.apply(void 0, n);
  };
var Fu = (
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
        var u = this.char();
        if (u === 123) {
          var l = this.parseArgument(t, n);
          if (l.err)
            return l;
          s.push(l.val);
        } else {
          if (u === 125 && t > 0)
            break;
          if (u === 35 && (r === "plural" || r === "selectordinal")) {
            var c = this.clonePosition();
            this.bump(), s.push({
              type: K.pound,
              location: U(c, this.clonePosition())
            });
          } else if (u === 60 && !this.ignoreTag && this.peek() === 47) {
            if (n)
              break;
            return this.error(D.UNMATCHED_CLOSING_TAG, U(this.clonePosition(), this.clonePosition()));
          } else if (u === 60 && !this.ignoreTag && vn(this.peek() || 0)) {
            var l = this.parseTag(t, r);
            if (l.err)
              return l;
            s.push(l.val);
          } else {
            var l = this.parseLiteral(t, r);
            if (l.err)
              return l;
            s.push(l.val);
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
            type: K.literal,
            value: "<".concat(s, "/>"),
            location: U(n, this.clonePosition())
          },
          err: null
        };
      if (this.bumpIf(">")) {
        var u = this.parseMessage(t + 1, r, !0);
        if (u.err)
          return u;
        var l = u.val, c = this.clonePosition();
        if (this.bumpIf("</")) {
          if (this.isEOF() || !vn(this.char()))
            return this.error(D.INVALID_TAG, U(c, this.clonePosition()));
          var d = this.clonePosition(), h = this.parseTagName();
          return s !== h ? this.error(D.UNMATCHED_CLOSING_TAG, U(d, this.clonePosition())) : (this.bumpSpace(), this.bumpIf(">") ? {
            val: {
              type: K.tag,
              value: s,
              children: l,
              location: U(n, this.clonePosition())
            },
            err: null
          } : this.error(D.INVALID_TAG, U(c, this.clonePosition())));
        } else
          return this.error(D.UNCLOSED_TAG, U(n, this.clonePosition()));
      } else
        return this.error(D.INVALID_TAG, U(n, this.clonePosition()));
    }, e.prototype.parseTagName = function() {
      var t = this.offset();
      for (this.bump(); !this.isEOF() && ju(this.char()); )
        this.bump();
      return this.message.slice(t, this.offset());
    }, e.prototype.parseLiteral = function(t, r) {
      for (var n = this.clonePosition(), s = ""; ; ) {
        var u = this.tryParseQuote(r);
        if (u) {
          s += u;
          continue;
        }
        var l = this.tryParseUnquoted(t, r);
        if (l) {
          s += l;
          continue;
        }
        var c = this.tryParseLeftAngleBracket();
        if (c) {
          s += c;
          continue;
        }
        break;
      }
      var d = U(n, this.clonePosition());
      return {
        val: { type: K.literal, value: s, location: d },
        err: null
      };
    }, e.prototype.tryParseLeftAngleBracket = function() {
      return !this.isEOF() && this.char() === 60 && (this.ignoreTag || // If at the opening tag or closing tag position, bail.
      !ku(this.peek() || 0)) ? (this.bump(), "<") : null;
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
      return dn.apply(void 0, r);
    }, e.prototype.tryParseUnquoted = function(t, r) {
      if (this.isEOF())
        return null;
      var n = this.char();
      return n === 60 || n === 123 || n === 35 && (r === "plural" || r === "selectordinal") || n === 125 && t > 0 ? null : (this.bump(), dn(n));
    }, e.prototype.parseArgument = function(t, r) {
      var n = this.clonePosition();
      if (this.bump(), this.bumpSpace(), this.isEOF())
        return this.error(D.EXPECT_ARGUMENT_CLOSING_BRACE, U(n, this.clonePosition()));
      if (this.char() === 125)
        return this.bump(), this.error(D.EMPTY_ARGUMENT, U(n, this.clonePosition()));
      var s = this.parseIdentifierIfPossible().value;
      if (!s)
        return this.error(D.MALFORMED_ARGUMENT, U(n, this.clonePosition()));
      if (this.bumpSpace(), this.isEOF())
        return this.error(D.EXPECT_ARGUMENT_CLOSING_BRACE, U(n, this.clonePosition()));
      switch (this.char()) {
        // Simple argument: `{name}`
        case 125:
          return this.bump(), {
            val: {
              type: K.argument,
              // value does not include the opening and closing braces.
              value: s,
              location: U(n, this.clonePosition())
            },
            err: null
          };
        // Argument with options: `{name, format, ...}`
        case 44:
          return this.bump(), this.bumpSpace(), this.isEOF() ? this.error(D.EXPECT_ARGUMENT_CLOSING_BRACE, U(n, this.clonePosition())) : this.parseArgumentOptions(t, r, s, n);
        default:
          return this.error(D.MALFORMED_ARGUMENT, U(n, this.clonePosition()));
      }
    }, e.prototype.parseIdentifierIfPossible = function() {
      var t = this.clonePosition(), r = this.offset(), n = pn(this.message, r), s = r + n.length;
      this.bumpTo(s);
      var u = this.clonePosition(), l = U(t, u);
      return { value: n, location: l };
    }, e.prototype.parseArgumentOptions = function(t, r, n, s) {
      var u, l = this.clonePosition(), c = this.parseIdentifierIfPossible().value, d = this.clonePosition();
      switch (c) {
        case "":
          return this.error(D.EXPECT_ARGUMENT_TYPE, U(l, d));
        case "number":
        case "date":
        case "time": {
          this.bumpSpace();
          var h = null;
          if (this.bumpIf(",")) {
            this.bumpSpace();
            var p = this.clonePosition(), E = this.parseSimpleArgStyleIfPossible();
            if (E.err)
              return E;
            var y = Uu(E.val);
            if (y.length === 0)
              return this.error(D.EXPECT_ARGUMENT_STYLE, U(this.clonePosition(), this.clonePosition()));
            var C = U(p, this.clonePosition());
            h = { style: y, styleLocation: C };
          }
          var I = this.tryParseArgumentClose(s);
          if (I.err)
            return I;
          var B = U(s, this.clonePosition());
          if (h && wi(h?.style, "::", 0)) {
            var O = Du(h.style.slice(2));
            if (c === "number") {
              var E = this.parseNumberSkeletonFromString(O, h.styleLocation);
              return E.err ? E : {
                val: { type: K.number, value: n, location: B, style: E.val },
                err: null
              };
            } else {
              if (O.length === 0)
                return this.error(D.EXPECT_DATE_TIME_SKELETON, B);
              var S = O;
              this.locale && (S = wu(O, this.locale));
              var y = {
                type: Ut.dateTime,
                pattern: S,
                location: h.styleLocation,
                parsedOptions: this.shouldParseSkeletons ? mu(S) : {}
              }, M = c === "date" ? K.date : K.time;
              return {
                val: { type: M, value: n, location: B, style: y },
                err: null
              };
            }
          }
          return {
            val: {
              type: c === "number" ? K.number : c === "date" ? K.date : K.time,
              value: n,
              location: B,
              style: (u = h?.style) !== null && u !== void 0 ? u : null
            },
            err: null
          };
        }
        case "plural":
        case "selectordinal":
        case "select": {
          var H = this.clonePosition();
          if (this.bumpSpace(), !this.bumpIf(","))
            return this.error(D.EXPECT_SELECT_ARGUMENT_OPTIONS, U(H, V({}, H)));
          this.bumpSpace();
          var _ = this.parseIdentifierIfPossible(), b = 0;
          if (c !== "select" && _.value === "offset") {
            if (!this.bumpIf(":"))
              return this.error(D.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, U(this.clonePosition(), this.clonePosition()));
            this.bumpSpace();
            var E = this.tryParseDecimalInteger(D.EXPECT_PLURAL_ARGUMENT_OFFSET_VALUE, D.INVALID_PLURAL_ARGUMENT_OFFSET_VALUE);
            if (E.err)
              return E;
            this.bumpSpace(), _ = this.parseIdentifierIfPossible(), b = E.val;
          }
          var A = this.tryParsePluralOrSelectOptions(t, c, r, _);
          if (A.err)
            return A;
          var I = this.tryParseArgumentClose(s);
          if (I.err)
            return I;
          var T = U(s, this.clonePosition());
          return c === "select" ? {
            val: {
              type: K.select,
              value: n,
              options: Ai(A.val),
              location: T
            },
            err: null
          } : {
            val: {
              type: K.plural,
              value: n,
              options: Ai(A.val),
              offset: b,
              pluralType: c === "plural" ? "cardinal" : "ordinal",
              location: T
            },
            err: null
          };
        }
        default:
          return this.error(D.INVALID_ARGUMENT_TYPE, U(l, d));
      }
    }, e.prototype.tryParseArgumentClose = function(t) {
      return this.isEOF() || this.char() !== 125 ? this.error(D.EXPECT_ARGUMENT_CLOSING_BRACE, U(t, this.clonePosition())) : (this.bump(), { val: !0, err: null });
    }, e.prototype.parseSimpleArgStyleIfPossible = function() {
      for (var t = 0, r = this.clonePosition(); !this.isEOF(); ) {
        var n = this.char();
        switch (n) {
          case 39: {
            this.bump();
            var s = this.clonePosition();
            if (!this.bumpUntil("'"))
              return this.error(D.UNCLOSED_QUOTE_IN_ARGUMENT_STYLE, U(s, this.clonePosition()));
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
        n = yu(t);
      } catch {
        return this.error(D.INVALID_NUMBER_SKELETON, r);
      }
      return {
        val: {
          type: Ut.number,
          tokens: n,
          location: r,
          parsedOptions: this.shouldParseSkeletons ? Tu(n) : {}
        },
        err: null
      };
    }, e.prototype.tryParsePluralOrSelectOptions = function(t, r, n, s) {
      for (var u, l = !1, c = [], d = /* @__PURE__ */ new Set(), h = s.value, p = s.location; ; ) {
        if (h.length === 0) {
          var E = this.clonePosition();
          if (r !== "select" && this.bumpIf("=")) {
            var y = this.tryParseDecimalInteger(D.EXPECT_PLURAL_ARGUMENT_SELECTOR, D.INVALID_PLURAL_ARGUMENT_SELECTOR);
            if (y.err)
              return y;
            p = U(E, this.clonePosition()), h = this.message.slice(E.offset, this.offset());
          } else
            break;
        }
        if (d.has(h))
          return this.error(r === "select" ? D.DUPLICATE_SELECT_ARGUMENT_SELECTOR : D.DUPLICATE_PLURAL_ARGUMENT_SELECTOR, p);
        h === "other" && (l = !0), this.bumpSpace();
        var C = this.clonePosition();
        if (!this.bumpIf("{"))
          return this.error(r === "select" ? D.EXPECT_SELECT_ARGUMENT_SELECTOR_FRAGMENT : D.EXPECT_PLURAL_ARGUMENT_SELECTOR_FRAGMENT, U(this.clonePosition(), this.clonePosition()));
        var I = this.parseMessage(t + 1, r, n);
        if (I.err)
          return I;
        var B = this.tryParseArgumentClose(C);
        if (B.err)
          return B;
        c.push([
          h,
          {
            value: I.val,
            location: U(C, this.clonePosition())
          }
        ]), d.add(h), this.bumpSpace(), u = this.parseIdentifierIfPossible(), h = u.value, p = u.location;
      }
      return c.length === 0 ? this.error(r === "select" ? D.EXPECT_SELECT_ARGUMENT_SELECTOR : D.EXPECT_PLURAL_ARGUMENT_SELECTOR, U(this.clonePosition(), this.clonePosition())) : this.requiresOtherClause && !l ? this.error(D.MISSING_OTHER_CLAUSE, U(this.clonePosition(), this.clonePosition())) : { val: c, err: null };
    }, e.prototype.tryParseDecimalInteger = function(t, r) {
      var n = 1, s = this.clonePosition();
      this.bumpIf("+") || this.bumpIf("-") && (n = -1);
      for (var u = !1, l = 0; !this.isEOF(); ) {
        var c = this.char();
        if (c >= 48 && c <= 57)
          u = !0, l = l * 10 + (c - 48), this.bump();
        else
          break;
      }
      var d = U(s, this.clonePosition());
      return u ? (l *= n, Ru(l) ? { val: l, err: null } : this.error(r, d)) : this.error(t, d);
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
      var r = ma(this.message, t);
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
function vn(e) {
  return e >= 97 && e <= 122 || e >= 65 && e <= 90;
}
function ku(e) {
  return vn(e) || e === 47;
}
function ju(e) {
  return e === 45 || e === 46 || e >= 48 && e <= 57 || e === 95 || e >= 97 && e <= 122 || e >= 65 && e <= 90 || e == 183 || e >= 192 && e <= 214 || e >= 216 && e <= 246 || e >= 248 && e <= 893 || e >= 895 && e <= 8191 || e >= 8204 && e <= 8205 || e >= 8255 && e <= 8256 || e >= 8304 && e <= 8591 || e >= 11264 && e <= 12271 || e >= 12289 && e <= 55295 || e >= 63744 && e <= 64975 || e >= 65008 && e <= 65533 || e >= 65536 && e <= 983039;
}
function ya(e) {
  return e >= 9 && e <= 13 || e === 32 || e === 133 || e >= 8206 && e <= 8207 || e === 8232 || e === 8233;
}
function Vu(e) {
  return e >= 33 && e <= 35 || e === 36 || e >= 37 && e <= 39 || e === 40 || e === 41 || e === 42 || e === 43 || e === 44 || e === 45 || e >= 46 && e <= 47 || e >= 58 && e <= 59 || e >= 60 && e <= 62 || e >= 63 && e <= 64 || e === 91 || e === 92 || e === 93 || e === 94 || e === 96 || e === 123 || e === 124 || e === 125 || e === 126 || e === 161 || e >= 162 && e <= 165 || e === 166 || e === 167 || e === 169 || e === 171 || e === 172 || e === 174 || e === 176 || e === 177 || e === 182 || e === 187 || e === 191 || e === 215 || e === 247 || e >= 8208 && e <= 8213 || e >= 8214 && e <= 8215 || e === 8216 || e === 8217 || e === 8218 || e >= 8219 && e <= 8220 || e === 8221 || e === 8222 || e === 8223 || e >= 8224 && e <= 8231 || e >= 8240 && e <= 8248 || e === 8249 || e === 8250 || e >= 8251 && e <= 8254 || e >= 8257 && e <= 8259 || e === 8260 || e === 8261 || e === 8262 || e >= 8263 && e <= 8273 || e === 8274 || e === 8275 || e >= 8277 && e <= 8286 || e >= 8592 && e <= 8596 || e >= 8597 && e <= 8601 || e >= 8602 && e <= 8603 || e >= 8604 && e <= 8607 || e === 8608 || e >= 8609 && e <= 8610 || e === 8611 || e >= 8612 && e <= 8613 || e === 8614 || e >= 8615 && e <= 8621 || e === 8622 || e >= 8623 && e <= 8653 || e >= 8654 && e <= 8655 || e >= 8656 && e <= 8657 || e === 8658 || e === 8659 || e === 8660 || e >= 8661 && e <= 8691 || e >= 8692 && e <= 8959 || e >= 8960 && e <= 8967 || e === 8968 || e === 8969 || e === 8970 || e === 8971 || e >= 8972 && e <= 8991 || e >= 8992 && e <= 8993 || e >= 8994 && e <= 9e3 || e === 9001 || e === 9002 || e >= 9003 && e <= 9083 || e === 9084 || e >= 9085 && e <= 9114 || e >= 9115 && e <= 9139 || e >= 9140 && e <= 9179 || e >= 9180 && e <= 9185 || e >= 9186 && e <= 9254 || e >= 9255 && e <= 9279 || e >= 9280 && e <= 9290 || e >= 9291 && e <= 9311 || e >= 9472 && e <= 9654 || e === 9655 || e >= 9656 && e <= 9664 || e === 9665 || e >= 9666 && e <= 9719 || e >= 9720 && e <= 9727 || e >= 9728 && e <= 9838 || e === 9839 || e >= 9840 && e <= 10087 || e === 10088 || e === 10089 || e === 10090 || e === 10091 || e === 10092 || e === 10093 || e === 10094 || e === 10095 || e === 10096 || e === 10097 || e === 10098 || e === 10099 || e === 10100 || e === 10101 || e >= 10132 && e <= 10175 || e >= 10176 && e <= 10180 || e === 10181 || e === 10182 || e >= 10183 && e <= 10213 || e === 10214 || e === 10215 || e === 10216 || e === 10217 || e === 10218 || e === 10219 || e === 10220 || e === 10221 || e === 10222 || e === 10223 || e >= 10224 && e <= 10239 || e >= 10240 && e <= 10495 || e >= 10496 && e <= 10626 || e === 10627 || e === 10628 || e === 10629 || e === 10630 || e === 10631 || e === 10632 || e === 10633 || e === 10634 || e === 10635 || e === 10636 || e === 10637 || e === 10638 || e === 10639 || e === 10640 || e === 10641 || e === 10642 || e === 10643 || e === 10644 || e === 10645 || e === 10646 || e === 10647 || e === 10648 || e >= 10649 && e <= 10711 || e === 10712 || e === 10713 || e === 10714 || e === 10715 || e >= 10716 && e <= 10747 || e === 10748 || e === 10749 || e >= 10750 && e <= 11007 || e >= 11008 && e <= 11055 || e >= 11056 && e <= 11076 || e >= 11077 && e <= 11078 || e >= 11079 && e <= 11084 || e >= 11085 && e <= 11123 || e >= 11124 && e <= 11125 || e >= 11126 && e <= 11157 || e === 11158 || e >= 11159 && e <= 11263 || e >= 11776 && e <= 11777 || e === 11778 || e === 11779 || e === 11780 || e === 11781 || e >= 11782 && e <= 11784 || e === 11785 || e === 11786 || e === 11787 || e === 11788 || e === 11789 || e >= 11790 && e <= 11798 || e === 11799 || e >= 11800 && e <= 11801 || e === 11802 || e === 11803 || e === 11804 || e === 11805 || e >= 11806 && e <= 11807 || e === 11808 || e === 11809 || e === 11810 || e === 11811 || e === 11812 || e === 11813 || e === 11814 || e === 11815 || e === 11816 || e === 11817 || e >= 11818 && e <= 11822 || e === 11823 || e >= 11824 && e <= 11833 || e >= 11834 && e <= 11835 || e >= 11836 && e <= 11839 || e === 11840 || e === 11841 || e === 11842 || e >= 11843 && e <= 11855 || e >= 11856 && e <= 11857 || e === 11858 || e >= 11859 && e <= 11903 || e >= 12289 && e <= 12291 || e === 12296 || e === 12297 || e === 12298 || e === 12299 || e === 12300 || e === 12301 || e === 12302 || e === 12303 || e === 12304 || e === 12305 || e >= 12306 && e <= 12307 || e === 12308 || e === 12309 || e === 12310 || e === 12311 || e === 12312 || e === 12313 || e === 12314 || e === 12315 || e === 12316 || e === 12317 || e >= 12318 && e <= 12319 || e === 12320 || e === 12336 || e === 64830 || e === 64831 || e >= 65093 && e <= 65094;
}
function gn(e) {
  e.forEach(function(t) {
    if (delete t.location, fa(t) || ca(t))
      for (var r in t.options)
        delete t.options[r].location, gn(t.options[r].value);
    else oa(t) && da(t.style) || (ua(t) || la(t)) && cn(t.style) ? delete t.style.location : ha(t) && gn(t.children);
  });
}
function Xu(e, t) {
  t === void 0 && (t = {}), t = V({ shouldParseSkeletons: !0, requiresOtherClause: !0 }, t);
  var r = new Fu(e, t).parse();
  if (r.err) {
    var n = SyntaxError(D[r.err.kind]);
    throw n.location = r.err.location, n.originalMessage = r.err.message, n;
  }
  return t?.captureLocation || gn(r.val), r.val;
}
var Ft;
(function(e) {
  e.MISSING_VALUE = "MISSING_VALUE", e.INVALID_VALUE = "INVALID_VALUE", e.MISSING_INTL_API = "MISSING_INTL_API";
})(Ft || (Ft = {}));
var Lr = (
  /** @class */
  (function(e) {
    Cr(t, e);
    function t(r, n, s) {
      var u = e.call(this, r) || this;
      return u.code = n, u.originalMessage = s, u;
    }
    return t.prototype.toString = function() {
      return "[formatjs Error: ".concat(this.code, "] ").concat(this.message);
    }, t;
  })(Error)
), Oi = (
  /** @class */
  (function(e) {
    Cr(t, e);
    function t(r, n, s, u) {
      return e.call(this, 'Invalid values for "'.concat(r, '": "').concat(n, '". Options are "').concat(Object.keys(s).join('", "'), '"'), Ft.INVALID_VALUE, u) || this;
    }
    return t;
  })(Lr)
), Wu = (
  /** @class */
  (function(e) {
    Cr(t, e);
    function t(r, n, s) {
      return e.call(this, 'Value for "'.concat(r, '" must be of type ').concat(n), Ft.INVALID_VALUE, s) || this;
    }
    return t;
  })(Lr)
), zu = (
  /** @class */
  (function(e) {
    Cr(t, e);
    function t(r, n) {
      return e.call(this, 'The intl string context variable "'.concat(r, '" was not provided to the string "').concat(n, '"'), Ft.MISSING_VALUE, n) || this;
    }
    return t;
  })(Lr)
), ve;
(function(e) {
  e[e.literal = 0] = "literal", e[e.object = 1] = "object";
})(ve || (ve = {}));
function Yu(e) {
  return e.length < 2 ? e : e.reduce(function(t, r) {
    var n = t[t.length - 1];
    return !n || n.type !== ve.literal || r.type !== ve.literal ? t.push(r) : n.value += r.value, t;
  }, []);
}
function qu(e) {
  return typeof e == "function";
}
function Sr(e, t, r, n, s, u, l) {
  if (e.length === 1 && xi(e[0]))
    return [
      {
        type: ve.literal,
        value: e[0].value
      }
    ];
  for (var c = [], d = 0, h = e; d < h.length; d++) {
    var p = h[d];
    if (xi(p)) {
      c.push({
        type: ve.literal,
        value: p.value
      });
      continue;
    }
    if (gu(p)) {
      typeof u == "number" && c.push({
        type: ve.literal,
        value: r.getNumberFormat(t).format(u)
      });
      continue;
    }
    var E = p.value;
    if (!(s && E in s))
      throw new zu(E, l);
    var y = s[E];
    if (vu(p)) {
      (!y || typeof y == "string" || typeof y == "number") && (y = typeof y == "string" || typeof y == "number" ? String(y) : ""), c.push({
        type: typeof y == "string" ? ve.literal : ve.object,
        value: y
      });
      continue;
    }
    if (ua(p)) {
      var C = typeof p.style == "string" ? n.date[p.style] : cn(p.style) ? p.style.parsedOptions : void 0;
      c.push({
        type: ve.literal,
        value: r.getDateTimeFormat(t, C).format(y)
      });
      continue;
    }
    if (la(p)) {
      var C = typeof p.style == "string" ? n.time[p.style] : cn(p.style) ? p.style.parsedOptions : n.time.medium;
      c.push({
        type: ve.literal,
        value: r.getDateTimeFormat(t, C).format(y)
      });
      continue;
    }
    if (oa(p)) {
      var C = typeof p.style == "string" ? n.number[p.style] : da(p.style) ? p.style.parsedOptions : void 0;
      C && C.scale && (y = y * (C.scale || 1)), c.push({
        type: ve.literal,
        value: r.getNumberFormat(t, C).format(y)
      });
      continue;
    }
    if (ha(p)) {
      var I = p.children, B = p.value, O = s[B];
      if (!qu(O))
        throw new Wu(B, "function", l);
      var S = Sr(I, t, r, n, s, u), M = O(S.map(function(b) {
        return b.value;
      }));
      Array.isArray(M) || (M = [M]), c.push.apply(c, M.map(function(b) {
        return {
          type: typeof b == "string" ? ve.literal : ve.object,
          value: b
        };
      }));
    }
    if (fa(p)) {
      var H = p.options[y] || p.options.other;
      if (!H)
        throw new Oi(p.value, y, Object.keys(p.options), l);
      c.push.apply(c, Sr(H.value, t, r, n, s));
      continue;
    }
    if (ca(p)) {
      var H = p.options["=".concat(y)];
      if (!H) {
        if (!Intl.PluralRules)
          throw new Lr(`Intl.PluralRules is not available in this environment.
Try polyfilling it using "@formatjs/intl-pluralrules"
`, Ft.MISSING_INTL_API, l);
        var _ = r.getPluralRules(t, { type: p.pluralType }).select(y - (p.offset || 0));
        H = p.options[_] || p.options.other;
      }
      if (!H)
        throw new Oi(p.value, y, Object.keys(p.options), l);
      c.push.apply(c, Sr(H.value, t, r, n, s, y - (p.offset || 0)));
      continue;
    }
  }
  return Yu(c);
}
function Zu(e, t) {
  return t ? V(V(V({}, e || {}), t || {}), Object.keys(e).reduce(function(r, n) {
    return r[n] = V(V({}, e[n]), t[n] || {}), r;
  }, {})) : e;
}
function Ju(e, t) {
  return t ? Object.keys(e).reduce(function(r, n) {
    return r[n] = Zu(e[n], t[n]), r;
  }, V({}, e)) : e;
}
function an(e) {
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
function $u(e) {
  return e === void 0 && (e = {
    number: {},
    dateTime: {},
    pluralRules: {}
  }), {
    getNumberFormat: tn(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.NumberFormat).bind.apply(t, en([void 0], r, !1)))();
    }, {
      cache: an(e.number),
      strategy: rn.variadic
    }),
    getDateTimeFormat: tn(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.DateTimeFormat).bind.apply(t, en([void 0], r, !1)))();
    }, {
      cache: an(e.dateTime),
      strategy: rn.variadic
    }),
    getPluralRules: tn(function() {
      for (var t, r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return new ((t = Intl.PluralRules).bind.apply(t, en([void 0], r, !1)))();
    }, {
      cache: an(e.pluralRules),
      strategy: rn.variadic
    })
  };
}
var Ku = (
  /** @class */
  (function() {
    function e(t, r, n, s) {
      r === void 0 && (r = e.defaultLocale);
      var u = this;
      if (this.formatterCache = {
        number: {},
        dateTime: {},
        pluralRules: {}
      }, this.format = function(d) {
        var h = u.formatToParts(d);
        if (h.length === 1)
          return h[0].value;
        var p = h.reduce(function(E, y) {
          return !E.length || y.type !== ve.literal || typeof E[E.length - 1] != "string" ? E.push(y.value) : E[E.length - 1] += y.value, E;
        }, []);
        return p.length <= 1 ? p[0] || "" : p;
      }, this.formatToParts = function(d) {
        return Sr(u.ast, u.locales, u.formatters, u.formats, d, void 0, u.message);
      }, this.resolvedOptions = function() {
        var d;
        return {
          locale: ((d = u.resolvedLocale) === null || d === void 0 ? void 0 : d.toString()) || Intl.NumberFormat.supportedLocalesOf(u.locales)[0]
        };
      }, this.getAst = function() {
        return u.ast;
      }, this.locales = r, this.resolvedLocale = e.resolveLocale(r), typeof t == "string") {
        if (this.message = t, !e.__parse)
          throw new TypeError("IntlMessageFormat.__parse must be set to process `message` of type `string`");
        var l = s || {};
        l.formatters;
        var c = ou(l, ["formatters"]);
        this.ast = e.__parse(t, V(V({}, c), { locale: this.resolvedLocale }));
      } else
        this.ast = t;
      if (!Array.isArray(this.ast))
        throw new TypeError("A message must be provided as a String or AST.");
      this.formats = Ju(e.formats, n), this.formatters = s && s.formatters || $u(this.formatterCache);
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
    }, e.__parse = Xu, e.formats = {
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
function Qu(e, t) {
  if (t == null)
    return;
  if (t in e)
    return e[t];
  const r = t.split(".");
  let n = e;
  for (let s = 0; s < r.length; s++)
    if (typeof n == "object") {
      if (s > 0) {
        const u = r.slice(s, r.length).join(".");
        if (u in n) {
          n = n[u];
          break;
        }
      }
      n = n[r[s]];
    } else
      n = void 0;
  return n;
}
const pt = {}, el = (e, t, r) => r && (t in pt || (pt[t] = {}), e in pt[t] || (pt[t][e] = r), r), xa = (e, t) => {
  if (t == null)
    return;
  if (t in pt && e in pt[t])
    return pt[t][e];
  const r = Rr(t);
  for (let n = 0; n < r.length; n++) {
    const s = r[n], u = rl(s, e);
    if (u)
      return el(e, t, u);
  }
};
let An;
const sr = Br({});
function tl(e) {
  return An[e] || null;
}
function Ea(e) {
  return e in An;
}
function rl(e, t) {
  if (!Ea(e))
    return null;
  const r = tl(e);
  return Qu(r, t);
}
function nl(e) {
  if (e == null)
    return;
  const t = Rr(e);
  for (let r = 0; r < t.length; r++) {
    const n = t[r];
    if (Ea(n))
      return n;
  }
}
function il(e, ...t) {
  delete pt[e], sr.update((r) => (r[e] = su.all([r[e] || {}, ...t]), r));
}
jt(
  [sr],
  ([e]) => Object.keys(e)
);
sr.subscribe((e) => An = e);
const Tr = {};
function al(e, t) {
  Tr[e].delete(t), Tr[e].size === 0 && delete Tr[e];
}
function Sa(e) {
  return Tr[e];
}
function sl(e) {
  return Rr(e).map((t) => {
    const r = Sa(t);
    return [t, r ? [...r] : []];
  }).filter(([, t]) => t.length > 0);
}
function bn(e) {
  return e == null ? !1 : Rr(e).some(
    (t) => {
      var r;
      return (r = Sa(t)) == null ? void 0 : r.size;
    }
  );
}
function ol(e, t) {
  return Promise.all(
    t.map((n) => (al(e, n), n().then((s) => s.default || s)))
  ).then((n) => il(e, ...n));
}
const er = {};
function Ta(e) {
  if (!bn(e))
    return e in er ? er[e] : Promise.resolve();
  const t = sl(e);
  return er[e] = Promise.all(
    t.map(
      ([r, n]) => ol(r, n)
    )
  ).then(() => {
    if (bn(e))
      return Ta(e);
    delete er[e];
  }), er[e];
}
const ul = {
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
}, ll = {
  fallbackLocale: null,
  loadingDelay: 200,
  formats: ul,
  warnOnMissingMessages: !0,
  handleMissingMessage: void 0,
  ignoreTag: !0
}, fl = ll;
function kt() {
  return fl;
}
const sn = Br(!1);
var cl = Object.defineProperty, hl = Object.defineProperties, dl = Object.getOwnPropertyDescriptors, Hi = Object.getOwnPropertySymbols, pl = Object.prototype.hasOwnProperty, vl = Object.prototype.propertyIsEnumerable, Ii = (e, t, r) => t in e ? cl(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, gl = (e, t) => {
  for (var r in t || (t = {}))
    pl.call(t, r) && Ii(e, r, t[r]);
  if (Hi)
    for (var r of Hi(t))
      vl.call(t, r) && Ii(e, r, t[r]);
  return e;
}, bl = (e, t) => hl(e, dl(t));
let mn;
const Hr = Br(null);
function Ni(e) {
  return e.split("-").map((t, r, n) => n.slice(0, r + 1).join("-")).reverse();
}
function Rr(e, t = kt().fallbackLocale) {
  const r = Ni(e);
  return t ? [.../* @__PURE__ */ new Set([...r, ...Ni(t)])] : r;
}
function At() {
  return mn ?? void 0;
}
Hr.subscribe((e) => {
  mn = e ?? void 0, typeof window < "u" && e != null && document.documentElement.setAttribute("lang", e);
});
const ml = (e) => {
  if (e && nl(e) && bn(e)) {
    const { loadingDelay: t } = kt();
    let r;
    return typeof window < "u" && At() != null && t ? r = window.setTimeout(
      () => sn.set(!0),
      t
    ) : sn.set(!0), Ta(e).then(() => {
      Hr.set(e);
    }).finally(() => {
      clearTimeout(r), sn.set(!1);
    });
  }
  return Hr.set(e);
}, Vt = bl(gl({}, Hr), {
  set: ml
}), Gr = (e) => {
  const t = /* @__PURE__ */ Object.create(null);
  return (n) => {
    const s = JSON.stringify(n);
    return s in t ? t[s] : t[s] = e(n);
  };
};
var _l = Object.defineProperty, Ir = Object.getOwnPropertySymbols, wa = Object.prototype.hasOwnProperty, Aa = Object.prototype.propertyIsEnumerable, Pi = (e, t, r) => t in e ? _l(e, t, { enumerable: !0, configurable: !0, writable: !0, value: r }) : e[t] = r, Mn = (e, t) => {
  for (var r in t || (t = {}))
    wa.call(t, r) && Pi(e, r, t[r]);
  if (Ir)
    for (var r of Ir(t))
      Aa.call(t, r) && Pi(e, r, t[r]);
  return e;
}, Xt = (e, t) => {
  var r = {};
  for (var n in e)
    wa.call(e, n) && t.indexOf(n) < 0 && (r[n] = e[n]);
  if (e != null && Ir)
    for (var n of Ir(e))
      t.indexOf(n) < 0 && Aa.call(e, n) && (r[n] = e[n]);
  return r;
};
const ar = (e, t) => {
  const { formats: r } = kt();
  if (e in r && t in r[e])
    return r[e][t];
  throw new Error(`[svelte-i18n] Unknown "${t}" ${e} format.`);
}, yl = Gr(
  (e) => {
    var t = e, { locale: r, format: n } = t, s = Xt(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format numbers');
    return n && (s = ar("number", n)), new Intl.NumberFormat(r, s);
  }
), xl = Gr(
  (e) => {
    var t = e, { locale: r, format: n } = t, s = Xt(t, ["locale", "format"]);
    if (r == null)
      throw new Error('[svelte-i18n] A "locale" must be set to format dates');
    return n ? s = ar("date", n) : Object.keys(s).length === 0 && (s = ar("date", "short")), new Intl.DateTimeFormat(r, s);
  }
), El = Gr(
  (e) => {
    var t = e, { locale: r, format: n } = t, s = Xt(t, ["locale", "format"]);
    if (r == null)
      throw new Error(
        '[svelte-i18n] A "locale" must be set to format time values'
      );
    return n ? s = ar("time", n) : Object.keys(s).length === 0 && (s = ar("time", "short")), new Intl.DateTimeFormat(r, s);
  }
), Sl = (e = {}) => {
  var t = e, {
    locale: r = At()
  } = t, n = Xt(t, [
    "locale"
  ]);
  return yl(Mn({ locale: r }, n));
}, Tl = (e = {}) => {
  var t = e, {
    locale: r = At()
  } = t, n = Xt(t, [
    "locale"
  ]);
  return xl(Mn({ locale: r }, n));
}, wl = (e = {}) => {
  var t = e, {
    locale: r = At()
  } = t, n = Xt(t, [
    "locale"
  ]);
  return El(Mn({ locale: r }, n));
}, Al = Gr(
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  (e, t = At()) => new Ku(e, t, kt().formats, {
    ignoreTag: kt().ignoreTag
  })
), Ml = (e, t = {}) => {
  var r, n, s, u;
  let l = t;
  typeof e == "object" && (l = e, e = l.id);
  const {
    values: c,
    locale: d = At(),
    default: h
  } = l;
  if (d == null)
    throw new Error(
      "[svelte-i18n] Cannot format a message without first setting the initial locale."
    );
  let p = xa(e, d);
  if (!p)
    p = (u = (s = (n = (r = kt()).handleMissingMessage) == null ? void 0 : n.call(r, { locale: d, id: e, defaultValue: h })) != null ? s : h) != null ? u : e;
  else if (typeof p != "string")
    return console.warn(
      `[svelte-i18n] Message with id "${e}" must be of type "string", found: "${typeof p}". Gettin its value through the "$format" method is deprecated; use the "json" method instead.`
    ), p;
  if (!c)
    return p;
  let E = p;
  try {
    E = Al(p, d).format(c);
  } catch (y) {
    y instanceof Error && console.warn(
      `[svelte-i18n] Message "${e}" has syntax error:`,
      y.message
    );
  }
  return E;
}, Ol = (e, t) => wl(t).format(e), Hl = (e, t) => Tl(t).format(e), Il = (e, t) => Sl(t).format(e), Nl = (e, t = At()) => xa(e, t);
jt([Vt, sr], () => Ml);
jt([Vt], () => Ol);
jt([Vt], () => Hl);
jt([Vt], () => Il);
jt([Vt, sr], () => Nl);
const Pl = "__i18n__", Bl = [
  "label",
  "info",
  "placeholder",
  "description",
  "title",
  "value"
], Cl = [
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
function Ll(e) {
  return typeof e == "string" && e.includes(Pl);
}
class Rl {
  load_component;
  #t = q(dt({}));
  get shared() {
    return a(this.#t);
  }
  set shared(t) {
    m(this.#t, t, !0);
  }
  #r = q(dt({}));
  get props() {
    return a(this.#r);
  }
  set props(t) {
    m(this.#r, t, !0);
  }
  #e = q((t) => t);
  get i18n() {
    return a(this.#e);
  }
  set i18n(t) {
    m(this.#e, t, !0);
  }
  translatable_props = {};
  dispatcher;
  last_update = null;
  shared_props = Cl;
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
    for (const n of Bl)
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
    ), Dt(() => {
      for (const n in t.shared_props)
        this._is_i18n_managed(`shared.${n}`, t.shared_props[n]) || (this.shared[n] = t.shared_props[n]);
      for (const n in t.props)
        this._is_i18n_managed(`props.${n}`, t.props[n]) || (this.props[n] = t.props[n]);
      this.register_component(
        t.shared_props.id,
        // @ts-ignore
        this.set_data.bind(this),
        this.get_data.bind(this)
      ), fe(() => {
        this.shared.id = t.shared_props.id;
      });
    }), Object.keys(this.translatable_props).length > 0 && Vt.subscribe(() => {
      for (const [n, s] of Object.entries(this.translatable_props)) {
        const [u, l] = n.split("."), c = this.i18n(s);
        u === "shared" ? this.shared[l] = c : this.props[l] = c;
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
    return Eo(this.props);
  }
  update(t) {
    this.set_data(t);
  }
  set_data(t) {
    for (const r in t) {
      const n = t[r], s = Ll(n) ? this._translate_and_store(this.shared_props.includes(r) ? "shared" : "props", r, n) : n;
      if (this.shared_props.includes(r)) {
        const u = r;
        this.shared[u] = s;
        continue;
      }
      this.props[r] = s;
    }
  }
  watch_for_change() {
    Dt(() => {
      this.mounted || (this.old_value = this.props.value, this.mounted = !0), this.old_value != this.props.value && (this.old_value = this.props.value, this.dispatch("change"));
    });
  }
}
mo();
var Gl = /* @__PURE__ */ Io('<svg class="resize-handle svelte-1stq1b1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><line x1="1" y1="9" x2="9" y2="1" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line><line x1="5" y1="9" x2="9" y2="5" stroke="gray" stroke-width="0.5" class="svelte-1stq1b1"></line></svg>'), Bi = /* @__PURE__ */ ye("<!> <!>", 1), Dl = /* @__PURE__ */ ye('<div class="placeholder svelte-1stq1b1"></div>');
function Ul(e, t) {
  Zi(t, !1);
  let r = Q(t, "height", 8, void 0), n = Q(t, "min_height", 8, void 0), s = Q(t, "max_height", 8, void 0), u = Q(t, "width", 8, void 0), l = Q(t, "elem_id", 8, ""), c = Q(t, "elem_classes", 24, () => []), d = Q(t, "variant", 8, "solid"), h = Q(t, "border_mode", 8, "base"), p = Q(t, "padding", 8, !0), E = Q(t, "type", 8, "normal"), y = Q(t, "test_id", 8, void 0), C = Q(t, "explicit_call", 8, !1), I = Q(t, "container", 8, !0), B = Q(t, "visible", 8, !0), O = Q(t, "allow_overflow", 8, !0), S = Q(t, "overflow_behavior", 8, "auto"), M = Q(t, "scale", 8, null), H = Q(t, "min_width", 8, 0), _ = Q(t, "flex", 12, !1), b = Q(t, "resizable", 8, !1), A = Q(t, "rtl", 8, !1), T = Q(t, "fullscreen", 12, !1), F = Q(t, "label", 8, void 0), ee = Ct(T()), G = Ct(), vt = E() === "fieldset" ? "fieldset" : "div", Ce = Ct(0), tt = Ct(0), ae = Ct(null);
  function Fe(te) {
    T() && te.key === "Escape" && T(!1);
  }
  const Ze = (te) => {
    if (te !== void 0) {
      if (typeof te == "number")
        return te + "px";
      if (typeof te == "string")
        return te;
    }
  }, ge = (te) => {
    let Oe = te.clientY;
    const xe = (rt) => {
      const Ee = rt.clientY - Oe;
      Oe = rt.clientY, yo(G, a(G).style.height = `${a(G).offsetHeight + Ee}px`);
    }, Je = () => {
      window.removeEventListener("mousemove", xe), window.removeEventListener("mouseup", Je);
    };
    window.addEventListener("mousemove", xe), window.addEventListener("mouseup", Je);
  };
  ci(
    () => (Be(T()), a(ee), a(G)),
    () => {
      T() !== a(ee) && (m(ee, T()), T() ? (m(ae, a(G).getBoundingClientRect()), m(Ce, a(G).offsetHeight), m(tt, a(G).offsetWidth), window.addEventListener("keydown", Fe)) : (m(ae, null), window.removeEventListener("keydown", Fe)));
    }
  ), ci(() => Be(B()), () => {
    B() || _(!1);
  }), _o(), Qo();
  var ke = No(), gt = Zr(ke);
  {
    var Le = (te) => {
      var Oe = Bi(), xe = Zr(Oe);
      Uo(xe, () => vt, !1, (Ee, Re) => {
        ia(Ee, (Ge) => m(G, Ge), () => a(G)), Ko(
          Ee,
          (Ge, De) => ({
            "data-testid": y(),
            id: l(),
            class: `block ${Ge ?? ""}`,
            dir: A() ? "rtl" : "ltr",
            "aria-label": F(),
            style: "",
            [rr]: {
              hidden: B() === "hidden",
              padded: p(),
              flex: _(),
              border_focus: h() === "focus",
              border_contrast: h() === "contrast",
              "hide-container": !C() && !I(),
              fullscreen: T(),
              animating: T() && a(ae) !== null,
              "auto-margin": M() === null
            },
            [Lt]: De
          }),
          [
            () => (Be(c()), fe(() => c()?.join(" ") || "")),
            () => ({
              height: (Be(T()), Be(r()), fe(() => T() ? void 0 : Ze(r()))),
              "min-height": (Be(T()), Be(n()), fe(() => T() ? void 0 : Ze(n()))),
              "max-height": (Be(T()), Be(s()), fe(() => T() ? void 0 : Ze(s()))),
              "--start-top": (a(ae), fe(() => a(ae) ? `${a(ae).top}px` : "0px")),
              "--start-left": (a(ae), fe(() => a(ae) ? `${a(ae).left}px` : "0px")),
              "--start-width": (a(ae), fe(() => a(ae) ? `${a(ae).width}px` : "0px")),
              "--start-height": (a(ae), fe(() => a(ae) ? `${a(ae).height}px` : "0px")),
              width: (Be(T()), Be(u()), fe(() => T() ? void 0 : typeof u() == "number" ? `calc(min(${u()}px, 100%))` : Ze(u()))),
              "border-style": d(),
              overflow: O() ? S() : "hidden",
              "flex-grow": M(),
              "min-width": `calc(min(${H()}px, 100%))`
            })
          ],
          void 0,
          void 0,
          "svelte-1stq1b1"
        );
        var He = Bi(), nt = Zr(He);
        Do(nt, t, "default", {});
        var it = z(nt, 2);
        {
          var Wt = (Ge) => {
            var De = Gl();
            j("mousedown", De, ge), le(Ge, De);
          };
          Ye(it, (Ge) => {
            b() && Ge(Wt);
          });
        }
        le(Re, He);
      });
      var Je = z(xe, 2);
      {
        var rt = (Ee) => {
          var Re = Dl();
          let He;
          et(() => He = Mr(Re, "", He, {
            height: a(Ce) + "px",
            width: a(tt) + "px"
          })), le(Ee, Re);
        };
        Ye(Je, (Ee) => {
          T() && Ee(rt);
        });
      }
      le(te, Oe);
    };
    Ye(gt, (te) => {
      (B() === !0 || B() === "hidden") && te(Le);
    });
  }
  le(e, ke), qi();
}
Tn(["click"]);
Tn(["click"]);
var Fl = /* @__PURE__ */ new Set(["$$slots", "$$events", "$$legacy"]), kl = /* @__PURE__ */ ye("<option> </option>"), jl = /* @__PURE__ */ ye('<span class="current-badge svelte-r41nsf">当前</span>'), Vl = /* @__PURE__ */ ye('<div role="button" tabindex="0"><span class="group-drag-handle svelte-r41nsf" title="拖动调整叠层顺序" aria-label="拖动调整叠层顺序">⠿</span> <span class="group-order svelte-r41nsf"></span> <span class="group-reorder-label svelte-r41nsf"> <!></span> <button type="button" class="visibility-button svelte-r41nsf"> </button> <button type="button" title="只显示此层">S</button></div>'), Xl = /* @__PURE__ */ ye('<div class="group-reorder svelte-r41nsf" aria-label="调整 Label 叠层顺序"><div class="group-reorder-hint svelte-r41nsf">拖动排序；眼睛与单层显示仅影响浏览器预览</div> <div class="group-reorder-list svelte-r41nsf"></div></div>'), Wl = /* @__PURE__ */ ye('<div class="group-picker svelte-r41nsf"><label for="layout-active-label" class="svelte-r41nsf">当前 Label</label> <select id="layout-active-label" class="svelte-r41nsf"></select> <div class="mode-switch svelte-r41nsf" aria-label="显示模式"><button type="button">对齐模式</button> <button type="button">交叠预览</button></div> <!></div>'), zl = /* @__PURE__ */ ye('<label class="svelte-r41nsf"><span> </span> <input type="number" class="svelte-r41nsf"/> <small class="svelte-r41nsf"> </small></label>'), Yl = /* @__PURE__ */ ye('<div class="numeric-toolbar svelte-r41nsf" aria-label="当前层精细参数"></div>'), ql = /* @__PURE__ */ ye('<button type="button" class="svelte-r41nsf"> </button>'), Zl = /* @__PURE__ */ ye('<div class="proposal-actions svelte-r41nsf"></div>'), Jl = /* @__PURE__ */ ye('<div><div class="proposal-copy svelte-r41nsf"><strong class="svelte-r41nsf"> </strong> <span> </span></div> <!></div>'), $l = /* @__PURE__ */ ye('<div class="overlap-legend svelte-r41nsf"><span class="legend-current svelte-r41nsf"> </span> <span class="legend-next svelte-r41nsf"> </span> <span class="legend-overlap svelte-r41nsf"> </span></div>'), Kl = /* @__PURE__ */ ye('<div class="layout-editor svelte-r41nsf"><!> <div class="toolbar svelte-r41nsf"><button type="button" class="svelte-r41nsf">居中归一</button> <button type="button" class="svelte-r41nsf">居中</button> <button type="button" class="svelte-r41nsf">适配</button> <button type="button" class="svelte-r41nsf">找回标注</button> <button type="button" class="svelte-r41nsf">画布 −</button> <button type="button" class="svelte-r41nsf"> </button> <button type="button" class="svelte-r41nsf">画布 ＋</button></div> <!> <!> <div class="canvas-wrap svelte-r41nsf"><canvas class="svelte-r41nsf"></canvas></div> <!> <div class="status svelte-r41nsf"><span> </span> <span class="svelte-r41nsf"> </span></div></div>');
function ef(e, t) {
  Zi(t, !0);
  const r = /* @__PURE__ */ tu(t, Fl), n = 2048, s = Math.log(1.01) / 100, u = 250, l = 1e4, c = [
    [0, 255, 120],
    [0, 188, 255],
    [255, 179, 0],
    [213, 94, 255],
    [255, 82, 82],
    [75, 222, 196]
  ], d = new Rl(r);
  let h, p = null, E = null, y = null, C = null, I = /* @__PURE__ */ new Map(), B = /* @__PURE__ */ new Map(), O = q(0), S = null, M = q(!1), H = q(!1), _ = q("等待版图 mask"), b = q(dt({ enabled: !1 })), A = q(dt(On())), T = q(""), F = q(!1), ee = q(!1), G = q("crosshair"), vt = "", Ce = 0, tt = { x: 0, y: 0, center_x: 0, center_y: 0 }, ae = { angle: 0, rotation: 0 }, Fe = null, Ze = null, ge = null, ke = null, gt = null, Le = 0, te = 0, Oe = "", xe = "", Je = "", rt = /* @__PURE__ */ new Set(), Ee = q(""), Re = q(""), He = q("before"), nt = q(!1), it = q("align"), Wt = q(dt(/* @__PURE__ */ new Set())), Ge = q(dt(/* @__PURE__ */ new Set())), De = q(""), ce = q(null), Dr = /* @__PURE__ */ new Set(), X = q(1), je = q(0), Ve = q(0), Mt = q(!1), or = q(!1), zt = { client_x: 0, client_y: 0, pan_x: 0, pan_y: 0 }, Yt = null, ur = q(""), at = q(dt({ x: "0", y: "0", scale: "100", rotation: "0", alpha: "22" }));
  function On() {
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
  function Hn(i) {
    return JSON.parse(JSON.stringify(i || { enabled: !1 }));
  }
  function Se(i) {
    const o = Object.assign(Object.assign({}, On()), i || {});
    return Object.assign(Object.assign({}, o), {
      center_x: ne(o.center_x, 0),
      center_y: ne(o.center_y, 0),
      pivot_x: ne(o.pivot_x, 0),
      pivot_y: ne(o.pivot_y, 0),
      scale: Te(ne(o.scale, 1), 0.01, 20),
      rotation_deg: lr(ne(o.rotation_deg, 0)),
      preview_alpha: Te(ne(o.preview_alpha, 0.35), 0, 1),
      revision: Math.max(0, Math.trunc(ne(o.revision, 0)))
    });
  }
  function ne(i, o) {
    const f = Number(i);
    return Number.isFinite(f) ? f : o;
  }
  function Ma(i) {
    return typeof i == "number" ? String(i) + "px" : i || "520px";
  }
  function Te(i, o, f) {
    return Math.max(o, Math.min(f, i));
  }
  function lr(i) {
    let o = ((i + 180) % 360 + 360) % 360 - 180;
    return o === -180 && (o = 180), o;
  }
  function k(i) {
    return typeof i + ":" + String(i);
  }
  function Xe() {
    var i;
    return Y() ? Array.isArray((i = a(b).group_view) === null || i === void 0 ? void 0 : i.groups) ? a(b).group_view.groups : [] : [];
  }
  function Ot() {
    var i;
    return Y() && ((i = a(b).group_view) === null || i === void 0 ? void 0 : i.allow_reorder) === !0;
  }
  function Y() {
    return a(b).transform_mode === "label_groups" && !!a(b).group_view;
  }
  function In() {
    for (const i of Xe())
      if (k(i.group_id) === a(T)) return i;
    return null;
  }
  function he() {
    const i = In();
    return i ? String(i.label || "Label " + String(i.group_id)) : "";
  }
  function qt() {
    const i = Xe(), o = i.findIndex((f) => k(f.group_id) === a(T));
    return o < 0 || o + 1 >= i.length ? "" : k(i[o + 1].group_id);
  }
  function Oa(i) {
    return a(De) ? i === a(De) : i === a(T) ? !0 : a(it) === "overlap" && i === qt();
  }
  function fr(i) {
    return a(Wt).has(i) ? !1 : a(Ge).has(i) ? !0 : Oa(i);
  }
  function Nn() {
    return Xe().map((i) => k(i.group_id)).filter((i) => fr(i));
  }
  function de() {
    return Math.max(1, Number(a(b).target_width || p?.naturalWidth || 1));
  }
  function we() {
    return Math.max(1, Number(a(b).target_height || p?.naturalHeight || 1));
  }
  function Ha() {
    var i;
    const o = (i = I.get(a(T))) === null || i === void 0 ? void 0 : i.image;
    return Math.max(1, Number(a(b).source_width || E?.naturalWidth || o?.naturalWidth || 1));
  }
  function Ia() {
    var i;
    const o = (i = I.get(a(T))) === null || i === void 0 ? void 0 : i.image;
    return Math.max(1, Number(a(b).source_height || E?.naturalHeight || o?.naturalHeight || 1));
  }
  function Na() {
    return Math.min(1, n / Math.max(de(), we()));
  }
  function Pn(i = a(T)) {
    var o;
    if (Y()) {
      const v = (o = I.get(i)) === null || o === void 0 ? void 0 : o.view.foreground_bbox_xyxy;
      if (Array.isArray(v) && v.length >= 4) return v.slice(0, 4).map(Number);
    }
    const f = a(b).foreground_bbox_xyxy;
    return Array.isArray(f) && f.length >= 4 ? f.slice(0, 4).map(Number) : [0, 0, Ha() - 1, Ia() - 1];
  }
  function Ur(i, o, f) {
    if (!i) {
      o === Ce && f(null);
      return;
    }
    const v = new Image();
    v.onload = () => {
      o === Ce && f(v);
    }, v.onerror = () => {
      o === Ce && f(null);
    }, v.src = i;
  }
  function Bn(i, o) {
    const f = Math.max(1, Number(a(b).source_width || i.naturalWidth || 1)), v = Math.max(1, Number(a(b).source_height || i.naturalHeight || 1)), g = document.createElement("canvas");
    g.width = f, g.height = v;
    const x = g.getContext("2d", { willReadFrequently: !0 });
    if (!x) return null;
    x.imageSmoothingEnabled = !1, x.drawImage(i, 0, 0, f, v);
    const w = x.getImageData(0, 0, f, v), N = document.createElement("canvas");
    N.width = f, N.height = v;
    const P = N.getContext("2d");
    if (!P) return null;
    const L = P.createImageData(f, v);
    for (let R = 0; R < w.data.length; R += 4) {
      const J = Math.max(w.data[R], w.data[R + 1], w.data[R + 2]);
      w.data[R + 3] > 0 && J >= 128 && (L.data[R] = o[0], L.data[R + 1] = o[1], L.data[R + 2] = o[2], L.data[R + 3] = 255);
    }
    return P.putImageData(L, 0, 0), { mask: g, tint: N };
  }
  function Pa() {
    if (!E) {
      y = null, C = null;
      return;
    }
    const i = Bn(E, c[0]);
    y = i?.mask || null, C = i?.tint || null;
  }
  function Ba(i) {
    if (!i.image) {
      i.maskCanvas = null, i.tintCanvas = null, i.ready = !1;
      return;
    }
    const o = Bn(i.image, i.color);
    i.maskCanvas = o?.mask || null, i.tintCanvas = o?.tint || null, i.ready = !!o;
  }
  function Ht() {
    Fe && (clearTimeout(Fe), Fe = null);
  }
  function It() {
    Ze && (clearTimeout(Ze), Ze = null);
  }
  function Ca(i) {
    const o = JSON.stringify(i);
    let f = 2166136261;
    for (let v = 0; v < o.length; v++)
      f ^= o.charCodeAt(v), f = Math.imul(f, 16777619);
    return "fnv1a32:" + (f >>> 0).toString(16).padStart(8, "0");
  }
  function pe() {
    Yt === null && (Yt = requestAnimationFrame(() => {
      Yt = null, Pt();
    }));
  }
  function Nt() {
    a(ur) || m(
      at,
      {
        x: ne(a(A).center_x, 0).toFixed(2).replace(/\.00$/, ""),
        y: ne(a(A).center_y, 0).toFixed(2).replace(/\.00$/, ""),
        scale: (ne(a(A).scale, 1) * 100).toFixed(3).replace(/0+$/, "").replace(/\.$/, ""),
        rotation: ne(a(A).rotation_deg, 0).toFixed(3).replace(/0+$/, "").replace(/\.$/, ""),
        alpha: (ne(a(A).preview_alpha, 0.22) * 100).toFixed(1).replace(/\.0$/, "")
      },
      !0
    );
  }
  function La(i, o) {
    if (!Array.isArray(o) || o.length !== i.length) return i;
    const f = /* @__PURE__ */ new Map();
    for (const x of i) {
      if (!x || x.group_id === void 0 || x.group_id === null) return i;
      const w = k(x.group_id);
      if (f.has(w)) return i;
      f.set(w, x);
    }
    const v = [], g = /* @__PURE__ */ new Set();
    for (const x of o) {
      if (typeof x != "string" && typeof x != "number") return i;
      const w = k(x), N = f.get(w);
      if (!N || g.has(w)) return i;
      g.add(w), v.push(N);
    }
    return v.length === i.length ? v : i;
  }
  function Ra(i) {
    var o;
    const f = a(b).group_view, v = Array.isArray(f?.groups) ? f.groups : [], g = a(b).group_intent, x = !!f && !!g && String(g.selection_signature || "") === String(f.selection_signature || "");
    Oe = String(a(b).editor_session_id || g?.editor_session_id || ""), xe = String(a(b).document_epoch || g?.document_epoch || f?.selection_signature || ""), Le = Math.max(0, Math.trunc(ne(
      (o = g?.client_revision) !== null && o !== void 0 ? o : g?.transform_set_revision,
      0
    ))), te = Le, ge = null, ke = null, It();
    const w = f?.allow_reorder === !0 && x ? La(v, g?.ordered_group_ids) : v;
    w !== v && f && m(
      b,
      Object.assign(Object.assign({}, a(b)), {
        group_view: Object.assign(Object.assign({}, f), { groups: w })
      }),
      !0
    );
    const N = /* @__PURE__ */ new Map();
    if (x && Array.isArray(g?.transforms))
      for (const Z of g.transforms)
        !Z || Z.group_id === void 0 || !Z.transform || N.set(k(Z.group_id), Se(Z.transform));
    const P = Se(a(b).transform), L = /* @__PURE__ */ new Map(), R = /* @__PURE__ */ new Map(), J = /* @__PURE__ */ new Set();
    for (let Z = 0; Z < w.length; Z++) {
      const Ie = w[Z];
      if (!Ie || Ie.group_id === void 0 || Ie.group_id === null) continue;
      const $e = k(Ie.group_id);
      if (J.has($e)) continue;
      J.add($e);
      const Ke = {
        view: Ie,
        image: null,
        maskCanvas: null,
        tintCanvas: null,
        ready: !1,
        color: c[Z % c.length]
      };
      L.set($e, Ke), R.set($e, Se(N.get($e) || P));
    }
    I = L, B = R, rt = /* @__PURE__ */ new Set(), Dr = /* @__PURE__ */ new Set(), m(ce, null);
    const W = x && g?.active_group_id !== void 0 && g.active_group_id !== null ? k(g.active_group_id) : "";
    m(
      T,
      L.has(W) ? W : L.keys().next().value || "",
      !0
    ), m(A, Se(B.get(a(T)) || P), !0), a(T) && B.set(a(T), a(A)), Nt(), E = null, y = null, C = null, m(H, !1);
    for (const [Z, Ie] of L)
      Ur(Ie.view.mask_image, i, ($e) => {
        const Ke = I.get(Z);
        !Ke || Ke !== Ie || (Ke.image = $e, Ba(Ke), m(O, a(O) + 1), pe());
      });
    m(
      _,
      a(b).status || (w.length > 0 ? "已加载 " + String(w.length) + " 个 Label；当前：" + he() : "当前选择没有可编辑 Label"),
      !0
    );
  }
  function Ga(i) {
    Ht(), It(), ge = null, ke = null, gt = null, Ce += 1;
    const o = Ce;
    m(b, Hn(i), !0), m(M, !1), I = /* @__PURE__ */ new Map(), B = /* @__PURE__ */ new Map(), m(T, ""), m(F, !1), m(ee, !1), Ur(a(b).base_image, o, (f) => {
      p = f, m(M, !!f), Pt();
    }), Y() ? Ra(o) : (m(A, Se(a(b).transform), !0), Nt(), m(_, a(b).status || "编辑器已加载", !0), m(H, !1), Ur(a(b).mask_image, o, (f) => {
      E = f, m(H, !!f), Pa(), Pt();
    }));
  }
  function Cn() {
    return Array.isArray(a(ce) === null || a(ce) === void 0 ? void 0 : a(ce).candidates) ? a(ce).candidates : [];
  }
  function Ln(i, o = !1) {
    if (!a(ce) || a(ce).target_layer_id === void 0) return;
    const f = k(a(ce).target_layer_id);
    if (!B.has(f)) return;
    if (o && (Dr.has(f) || ne(We(f).revision, 0) !== ne(a(ce).start_transform_revision, 0))) {
      m(_, "粗对齐结果已到达，但检测到手动修改，未覆盖当前变换"), st();
      return;
    }
    $t(f);
    const v = Se(We(f));
    ze(Object.assign(Object.assign(Object.assign({}, v), i.transform), { preview_alpha: v.preview_alpha }), !1), m(ce, null), pr("boundary_alignment", (o ? "已自动应用 " : "已应用 ") + String(i.candidate_id || "候选") + "；停止 250ms 后同步");
  }
  function Da(i) {
    const o = String(i.editor_session_id || ""), f = String(i.document_epoch || "");
    if (o && o !== Oe || f && f !== xe) return;
    m(ce, Hn(i), !0), m(_, String(i.status || "粗对齐候选已生成"), !0), st();
    const v = Array.isArray(i.candidates) ? i.candidates : [];
    i.auto_apply && v.length ? Ln(v[0], !0) : pe();
  }
  function Ua(i) {
    var o;
    const f = String(i.editor_session_id || ""), v = String(i.document_epoch || "");
    if (f !== Oe || v !== xe)
      return;
    if (i.payload_kind === "ack_error") {
      It(), ge = null, ke = null, m(_, String(i.status || "同步失败；本地修改仍保留"), !0), st(), Pt();
      return;
    }
    const g = Math.max(0, Math.trunc(ne(i.server_ack_revision, 0))), x = String(i.payload_hash || "");
    ge !== null && (g !== ge || x !== ke) || ge === null && (g !== Le || x !== String(((o = a(b).group_intent) === null || o === void 0 ? void 0 : o.payload_hash) || "")) || (It(), te = Math.max(te, g), gt = null, ge = null, ke = null, m(_, Le > te ? "本地修改待同步" : "已同步", !0), st(), Le > te && hr(0), Pt());
  }
  Dt(() => {
    const i = d.props.value || null, o = JSON.stringify(i);
    if (i?.payload_kind === "ack" || i?.payload_kind === "ack_error") {
      o !== Je && (Je = o, Ua(i));
      return;
    }
    if (i?.payload_kind === "alignment_proposal") {
      o !== Je && (Je = o, Da(i));
      return;
    }
    o !== vt && (vt = o, Ga(i));
  }), Bo(() => {
    Ce += 1, Ht(), It(), Yt !== null && cancelAnimationFrame(Yt);
  });
  function Fr(i) {
    const o = Number(i.rotation_deg || 0) * Math.PI / 180, f = Number(i.scale || 1), v = Math.cos(o), g = Math.sin(o), x = f * v, w = f * g, N = Number(i.center_x || 0) - x * Number(i.pivot_x || 0) + w * Number(i.pivot_y || 0), P = Number(i.center_y || 0) - w * Number(i.pivot_x || 0) - x * Number(i.pivot_y || 0);
    return [x, w, -w, x, N, P];
  }
  function Zt(i, o, f = a(A)) {
    const [v, g, x, w, N, P] = Fr(f);
    return { x: v * i + x * o + N, y: g * i + w * o + P };
  }
  function kr(i, o, f = a(A)) {
    const [v, g, x, w, N, P] = Fr(f), L = v * w - g * x;
    if (Math.abs(L) < 1e-9) return { x: -1, y: -1 };
    const R = i - N, J = o - P;
    return { x: (w * R - x * J) / L, y: (-g * R + v * J) / L };
  }
  function Jt(i) {
    const o = h.getBoundingClientRect(), f = (i.clientX - o.left) / Math.max(1, o.width) * de(), v = (i.clientY - o.top) / Math.max(1, o.height) * we();
    return {
      x: (f - a(je)) / Math.max(0.05, a(X)),
      y: (v - a(Ve)) / Math.max(0.05, a(X))
    };
  }
  function jr(i, o) {
    i.setTransform(o * a(X), 0, 0, o * a(X), o * a(je), o * a(Ve));
  }
  function Rn(i, o, f) {
    if (!i) return !1;
    const v = Math.round(o), g = Math.round(f);
    if (v < 0 || g < 0 || v >= i.width || g >= i.height) return !1;
    const x = i.getContext("2d", { willReadFrequently: !0 });
    if (!x) return !1;
    const w = x.getImageData(v, g, 1, 1).data;
    return w[3] > 0 && Math.max(w[0], w[1], w[2]) >= 128;
  }
  function We(i) {
    return !Y() || i === a(T) ? a(A) : B.get(i) || a(A);
  }
  function ze(i, o = !0) {
    m(A, Se(i), !0), Y() && a(T) && (B.set(a(T), a(A)), o && Dr.add(a(T))), Nt();
  }
  function Gn(i, o) {
    const f = kr(i, o, a(A));
    return Rn(y, f.x, f.y);
  }
  function Dn(i, o, f) {
    const v = I.get(i);
    if (!v?.ready) return !1;
    const g = kr(o, f, We(i));
    return Rn(v.maskCanvas, g.x, g.y);
  }
  function Vr(i, o) {
    const f = new Set(Nn());
    if (a(T) && f.has(a(T)) && Dn(a(T), i, o)) return a(T);
    const v = Xe().map((g) => k(g.group_id)).reverse();
    for (const g of v)
      if (g !== a(T) && f.has(g) && Dn(g, i, o)) return g;
    return "";
  }
  function cr(i = a(T), o = We(i)) {
    const [f, v, g, x] = Pn(i);
    return [
      Zt(f, v, o),
      Zt(g, v, o),
      Zt(g, x, o),
      Zt(f, x, o)
    ];
  }
  function Un() {
    const i = cr();
    let o = i[0];
    for (const w of i)
      (w.y < o.y || Math.abs(w.y - o.y) < 1e-6 && w.x > o.x) && (o = w);
    const f = Number(a(A).rotation_deg || 0) * Math.PI / 180, v = Math.cos(f - Math.PI / 4), g = Math.sin(f - Math.PI / 4), x = 34 / Math.max(0.05, a(X));
    return { x: o.x + v * x, y: o.y + g * x };
  }
  function Fn() {
    const i = h?.getBoundingClientRect();
    return i ? Math.max(8 / a(X), 14 * de() / Math.max(1, i.width) / a(X)) : 14 / a(X);
  }
  function kn(i, o) {
    if (Y() && !a(T)) return !1;
    const f = Un(), v = Fn();
    return Math.hypot(i - f.x, o - f.y) <= v;
  }
  function jn(i) {
    const o = Y() ? a(T) : "";
    ze(Object.assign(Object.assign({}, a(A)), {
      revision: Number(a(A).revision || 0) + 1,
      origin: i,
      scale: Te(Number(a(A).scale || 1), 0.01, 20),
      rotation_deg: lr(Number(a(A).rotation_deg || 0))
    })), o && rt.add(o);
  }
  function Fa() {
    const i = [];
    for (const o of Xe()) {
      const f = k(o.group_id);
      i.push({
        group_id: o.group_id,
        transform: Object.assign({}, Se(We(f)))
      });
    }
    return i;
  }
  function st() {
    var i;
    const o = {
      payload_kind: "intent",
      editor_session_id: Oe,
      document_epoch: xe,
      payload_hash: String(((i = a(b).group_intent) === null || i === void 0 ? void 0 : i.payload_hash) || ""),
      enabled: a(b).enabled,
      transform: Object.assign({}, Se(a(A))),
      target_width: a(b).target_width,
      target_height: a(b).target_height
    };
    Y() && (o.transform_mode = "label_groups", o.group_intent = a(b).group_intent ? JSON.parse(JSON.stringify(a(b).group_intent)) : null), d.props.value = o, vt = JSON.stringify(o);
  }
  function ka() {
    if (!Y()) {
      d.dispatch("change");
      return;
    }
    const i = a(b).group_intent, o = Math.max(0, Math.trunc(ne(i?.client_revision, 0)));
    !i || o <= te || ge !== null || (It(), ge = o, ke = String(i.payload_hash || ""), m(_, "同步中…"), st(), d.dispatch("change"), Ze = setTimeout(
      () => {
        if (ge !== o) return;
        ge = null, ke = null;
        const f = gt !== o;
        gt = o, m(_, f ? "同步超时，正在重试；本地修改仍保留" : "同步失败；可继续编辑或直接保存重试", !0), st(), Pt(), f && hr(1e3);
      },
      l
    ));
  }
  function hr(i = u) {
    Ht(), Fe = setTimeout(
      () => {
        Fe = null, ka();
      },
      Math.max(0, i)
    );
  }
  function dr(i, o, f, v = !0) {
    var g;
    if (!Y()) return;
    f && jn(i);
    const x = a(b).group_view, w = In(), N = Xe().filter((R) => rt.has(k(R.group_id))).map((R) => R.group_id), P = Le + 1, L = Object.assign(
      Object.assign(
        {
          selection_signature: String(x?.selection_signature || xe),
          transform_set_revision: P,
          editor_session_id: Oe,
          document_epoch: xe,
          client_revision: P,
          payload_hash: "",
          active_group_id: (g = w?.group_id) !== null && g !== void 0 ? g : null,
          changed_group_ids: N
        },
        Ot() ? {
          ordered_group_ids: Xe().map((R) => R.group_id)
        } : {}
      ),
      { transforms: Fa() }
    );
    L.payload_hash = Ca(Object.assign(Object.assign({}, L), { payload_hash: "" })), Le = P, gt = null, m(
      b,
      Object.assign(Object.assign({}, a(b)), {
        payload_kind: "intent",
        editor_session_id: Oe,
        document_epoch: xe,
        payload_hash: L.payload_hash,
        group_intent: L
      }),
      !0
    ), m(_, o + "（本地修改）"), st(), v && hr(0), pe();
  }
  function bt(i, o, f = !0) {
    if (Y()) {
      dr(i, o, !0, f);
      return;
    }
    jn(i), m(
      b,
      Object.assign(Object.assign({}, a(b)), {
        enabled: !0,
        transform: Object.assign({}, a(A)),
        status: o
      }),
      !0
    ), m(_, o, !0), st(), f && (Ht(), d.dispatch("change")), pe();
  }
  function pr(i, o, f = u) {
    if (Y()) {
      bt(i, o, !1), hr(f);
      return;
    }
    bt(i, o, !1), Ht(), Fe = setTimeout(
      () => {
        Fe = null, d.dispatch("change");
      },
      f
    );
  }
  function ja(i, o, f) {
    if (p && a(M)) {
      i.imageSmoothingEnabled = !0, i.drawImage(p, 0, 0, o, f);
      return;
    }
    i.fillStyle = "#f8fafc", i.fillRect(0, 0, o, f), i.fillStyle = "#64748b", i.font = "18px sans-serif", i.fillText("请先上传图像", 24, 42);
  }
  function vr(i, o, f, v, g) {
    var x;
    g === void 0 && (g = Te(Number((x = v.preview_alpha) !== null && x !== void 0 ? x : 0.35), 0, 1)), i.save(), i.globalAlpha = g;
    const [w, N, P, L, R, J] = Fr(v);
    i.setTransform(o * a(X) * w, o * a(X) * N, o * a(X) * P, o * a(X) * L, o * (a(X) * R + a(je)), o * (a(X) * J + a(Ve))), i.imageSmoothingEnabled = !1, i.drawImage(f, 0, 0, f.width, f.height), i.restore();
  }
  function Va(i, o, f, v) {
    const g = I.get(f), x = I.get(v);
    if (!g?.tintCanvas || !x?.tintCanvas || !g.ready || !x.ready) return;
    const w = S || document.createElement("canvas");
    S = w, w.width !== h.width && (w.width = h.width), w.height !== h.height && (w.height = h.height);
    const N = w.getContext("2d");
    if (N) {
      N.setTransform(1, 0, 0, 1, 0, 0), N.globalCompositeOperation = "source-over", N.clearRect(0, 0, w.width, w.height), vr(N, o, g.tintCanvas, We(f), 1), N.globalCompositeOperation = "destination-in", vr(N, o, x.tintCanvas, We(v), 1), N.globalCompositeOperation = "source-in", N.setTransform(1, 0, 0, 1, 0, 0), N.fillStyle = "#ff3fa4", N.fillRect(0, 0, w.width, w.height), N.globalCompositeOperation = "source-atop", N.strokeStyle = "rgba(255,255,255,0.78)", N.lineWidth = 1.5;
      for (let P = -w.height; P < w.width; P += 14)
        N.beginPath(), N.moveTo(P, 0), N.lineTo(P + w.height, w.height), N.stroke();
      i.save(), i.setTransform(1, 0, 0, 1, 0, 0), i.globalAlpha = 0.62, i.drawImage(w, 0, 0), i.restore();
    }
  }
  function Vn(i, o) {
    const f = cr();
    i.save(), i.lineJoin = "round", i.lineWidth = Math.max(2.5, o / 700) / a(X), i.strokeStyle = "rgba(0,0,0,0.82)", i.beginPath(), i.moveTo(f[0].x, f[0].y);
    for (let x = 1; x < f.length; x++) i.lineTo(f[x].x, f[x].y);
    i.closePath(), i.stroke(), i.lineWidth = Math.max(1.8, o / 1e3) / a(X), i.strokeStyle = "#00ff66", i.stroke();
    const v = Un(), g = f.reduce((x, w) => w.y < x.y || Math.abs(w.y - x.y) < 1e-6 && w.x > x.x ? w : x, f[0]);
    i.strokeStyle = "#0f172a", i.lineWidth = Math.max(2, o / 900) / a(X), i.beginPath(), i.moveTo(g.x, g.y), i.lineTo(v.x, v.y), i.stroke(), i.fillStyle = a(ee) ? "#ffb000" : "#ffffff", i.strokeStyle = "#00ff66", i.lineWidth = Math.max(2, o / 900) / a(X), i.beginPath(), i.arc(v.x, v.y, Fn(), 0, Math.PI * 2), i.fill(), i.stroke(), i.restore();
  }
  function Xa(i, o, f) {
    const v = I.get(o);
    if (!v) return;
    const g = cr(o, We(o)), x = Math.min(...g.map((W) => W.x)), w = Math.min(...g.map((W) => W.y)), N = String(v.view.label || "Label " + String(v.view.group_id)), P = 1 / Math.max(0.05, a(X));
    i.save(), i.font = "600 " + String(13 * P) + "px sans-serif";
    const L = i.measureText(N).width + 12 * P, R = Te(x, 2, Math.max(2, de() - L - 2)), J = Te(w - 23 * P, 2, Math.max(2, we() - 22 * P));
    i.fillStyle = f ? "rgba(15,23,42,0.92)" : "rgba(51,65,85,0.62)", i.fillRect(R, J, L, 20 * P), i.fillStyle = "#ffffff", i.fillText(N, R + 6 * P, J + 14 * P), i.restore();
  }
  function Pt() {
    var i, o, f;
    if (!h) return;
    const v = de(), g = we(), x = Na(), w = Math.max(1, Math.round(v * x)), N = Math.max(1, Math.round(g * x));
    h.width !== w && (h.width = w), h.height !== N && (h.height = N);
    const P = h.getContext("2d");
    if (P && (P.setTransform(1, 0, 0, 1, 0, 0), P.clearRect(0, 0, h.width, h.height), jr(P, x), ja(P, v, g), a(b).enabled !== !1)) {
      if (Y()) {
        const L = Nn(), R = L.filter((W) => W !== a(T));
        L.includes(a(T)) && R.push(a(T));
        const J = qt();
        for (const W of R) {
          const Z = I.get(W);
          if (Z?.ready && Z.tintCanvas) {
            const Ie = W === a(T) ? Te(Number((i = We(W).preview_alpha) !== null && i !== void 0 ? i : 0.22), 0, 1) : W === J ? 0.1 : 0.08;
            vr(P, x, Z.tintCanvas, We(W), Ie);
          }
        }
        a(it) === "overlap" && a(T) && J && L.includes(a(T)) && L.includes(J) && Va(P, x, a(T), J), jr(P, x);
        for (const W of R)
          !((o = I.get(W)) === null || o === void 0) && o.ready && Xa(P, W, W === a(T));
        a(T) && L.includes(a(T)) && (!((f = I.get(a(T))) === null || f === void 0) && f.ready) && Vn(P, v);
        return;
      }
      C && a(H) && (vr(P, x, C, a(A)), jr(P, x), Vn(P, v));
    }
  }
  function be() {
    var i;
    return a(b).enabled ? Y() ? (a(O), !!a(T) && !!(!((i = I.get(a(T))) === null || i === void 0) && i.ready)) : a(H) && !!y : !1;
  }
  function $t(i) {
    return !Y() || !I.has(i) || i === a(T) ? !1 : (a(T) && B.set(a(T), Se(a(A))), m(T, i, !0), m(A, Se(B.get(i)), !0), B.set(i, a(A)), Nt(), m(_, "当前 Label：" + he()), pe(), !0);
  }
  function Xn(i) {
    m(it, i, !0), m(_, i === "align" ? "对齐模式：默认只显示当前层" : "交叠预览：当前层突出、下一业务层淡显", !0), pe();
  }
  function Wa(i, o) {
    i.stopPropagation();
    const f = new Set(a(Wt)), v = new Set(a(Ge));
    fr(o) ? (f.add(o), v.delete(o)) : (f.delete(o), v.add(o)), m(Wt, f, !0), m(Ge, v, !0), m(_, "显示设置仅保存在浏览器，不改变保存数据"), pe();
  }
  function za(i, o) {
    i.stopPropagation(), m(De, a(De) === o ? "" : o, !0), m(_, a(De) ? "已单独显示一层" : "已退出单层显示", !0), pe();
  }
  function Ya() {
    m(X, 1), m(je, 0), m(Ve, 0), m(_, "画布视图已复位（不影响保存数据）"), pe();
  }
  function Wn(i) {
    const o = Te(a(X) * i, 0.2, 8), f = de() / 2, v = we() / 2, g = (f - a(je)) / a(X), x = (v - a(Ve)) / a(X);
    m(je, f - g * o), m(Ve, v - x * o), m(X, o, !0), m(_, "画布视图 " + Math.round(a(X) * 100) + "%（仅本地）"), pe();
  }
  function qa(i, o) {
    return i === "x" ? Object.assign(Object.assign({}, a(A)), { center_x: o }) : i === "y" ? Object.assign(Object.assign({}, a(A)), { center_y: o }) : i === "scale" ? Object.assign(Object.assign({}, a(A)), { scale: Te(o / 100, 0.01, 20) }) : i === "rotation" ? Object.assign(Object.assign({}, a(A)), { rotation_deg: lr(o) }) : Object.assign(Object.assign({}, a(A)), { preview_alpha: Te(o / 100, 0, 1) });
  }
  function Xr(i) {
    const o = Number(a(at)[i]);
    if (m(ur, ""), !Number.isFinite(o)) {
      Nt();
      return;
    }
    const f = Se(a(A)), v = Se(qa(i, o)), g = Math.abs(v.center_x - f.center_x) < 1e-9 && Math.abs(v.center_y - f.center_y) < 1e-9 && Math.abs(v.scale - f.scale) < 1e-12 && Math.abs(v.rotation_deg - f.rotation_deg) < 1e-9 && Math.abs(v.preview_alpha - f.preview_alpha) < 1e-12;
    if (ze(v), g) {
      pe();
      return;
    }
    pr("numeric", he() + " 参数已更新");
  }
  function Za(i, o) {
    m(at, Object.assign(Object.assign({}, a(at)), { [o]: i.currentTarget.value }), !0);
  }
  function Ja(i, o) {
    const f = o.key === "ArrowUp" ? 1 : -1, v = i === "x" || i === "y" || i === "alpha" ? 1 : 0.1, g = i === "x" || i === "y" || i === "alpha" ? 0.1 : 0.01, x = i === "x" || i === "y" ? 10 : i === "alpha" ? 5 : 1;
    return f * (o.shiftKey ? g : o.altKey ? x : v);
  }
  function $a(i, o) {
    if (i.key === "Escape") {
      i.preventDefault(), m(ur, ""), Nt(), i.currentTarget.blur();
      return;
    }
    if (i.key === "Enter") {
      i.preventDefault(), Xr(o), i.currentTarget.blur();
      return;
    }
    if (i.key === "ArrowUp" || i.key === "ArrowDown") {
      i.preventDefault();
      const f = ne(a(at)[o], 0);
      m(at, Object.assign(Object.assign({}, a(at)), { [o]: String(f + Ja(o, i)) }), !0), Xr(o);
    }
  }
  function Ka(i) {
    i.code === "Space" && !(i.target instanceof HTMLInputElement) && !(i.target instanceof HTMLTextAreaElement) && m(or, !0);
  }
  function Qa(i) {
    i.code === "Space" && m(or, !1);
  }
  function zn() {
    m(Ee, ""), m(Re, ""), m(He, "before");
  }
  function es(i, o) {
    Ot() && (m(Ee, o, !0), m(Re, ""), m(He, "before"), m(nt, !1), i.dataTransfer && (i.dataTransfer.effectAllowed = "move", i.dataTransfer.setData("text/plain", o)));
  }
  function ts(i, o) {
    if (!Ot() || !a(Ee)) return;
    i.preventDefault(), i.dataTransfer && (i.dataTransfer.dropEffect = "move");
    const v = i.currentTarget.getBoundingClientRect();
    m(Re, o, !0), m(He, i.clientY < v.top + v.height / 2 ? "before" : "after", !0);
  }
  function rs(i, o) {
    var f;
    if (!Ot()) return;
    i.preventDefault();
    const v = a(Ee) || ((f = i.dataTransfer) === null || f === void 0 ? void 0 : f.getData("text/plain")) || "", g = a(Re) === o ? a(He) : "before", x = is(v, o, g);
    zn(), x && (m(nt, !0), dr("reorder_groups", "已调整叠层顺序；拖动调整叠层顺序", !1, !0), setTimeout(
      () => {
        m(nt, !1);
      },
      0
    ));
  }
  function ns() {
    zn();
  }
  function is(i, o, f) {
    if (!Ot() || i === o) return !1;
    const v = a(b).group_view, g = Xe(), x = g.findIndex((W) => k(W.group_id) === i), w = g.findIndex((W) => k(W.group_id) === o);
    if (!v || x < 0 || w < 0) return !1;
    const N = g.slice(), [P] = N.splice(x, 1);
    if (!P) return !1;
    const L = N.findIndex((W) => k(W.group_id) === o);
    if (L < 0) return !1;
    const R = L + (f === "after" ? 1 : 0);
    return N.splice(R, 0, P), N.some((W, Z) => k(W.group_id) !== k(g[Z].group_id)) ? (m(
      b,
      Object.assign(Object.assign({}, a(b)), {
        group_view: Object.assign(Object.assign({}, v), { groups: N })
      }),
      !0
    ), !0) : !1;
  }
  function Yn(i, o) {
    if (a(nt)) {
      m(nt, !1);
      return;
    }
    const f = i.target;
    f?.closest(".group-drag-handle") || $t(o) && dr("select_group", "已选择 Label：" + he(), !1, !0);
  }
  function as(i) {
    const o = i.currentTarget.value;
    $t(o) && dr("select_group", "已选择 Label：" + he(), !1, !0);
  }
  function qn(i) {
    m(Mt, !0), m(G, "grabbing"), zt = {
      client_x: i.clientX,
      client_y: i.clientY,
      pan_x: a(je),
      pan_y: a(Ve)
    }, h.setPointerCapture(i.pointerId);
  }
  function ss(i) {
    if (i.button !== 0) return;
    Ht();
    const o = Jt(i);
    if (a(or)) {
      qn(i);
      return;
    }
    if (be() && kn(o.x, o.y)) {
      m(G, "grabbing"), m(ee, !0), ae = {
        angle: Math.atan2(o.y - a(A).center_y, o.x - a(A).center_x) * 180 / Math.PI,
        rotation: Number(a(A).rotation_deg || 0)
      }, h.setPointerCapture(i.pointerId);
      return;
    }
    let f = !1;
    if (be() && Y()) {
      const v = Vr(o.x, o.y);
      v && ($t(v), f = !0);
    } else be() && (f = Gn(o.x, o.y));
    if (!f) {
      qn(i), m(_, "正在平移画布视图（仅本地）");
      return;
    }
    m(G, "grabbing"), m(F, !0), tt = {
      x: o.x,
      y: o.y,
      center_x: Number(a(A).center_x || 0),
      center_y: Number(a(A).center_y || 0)
    }, h.setPointerCapture(i.pointerId);
  }
  function Wr(i) {
    if (a(or)) {
      m(G, "grab");
      return;
    }
    if (!be()) {
      m(G, "grab");
      return;
    }
    const o = Y() ? !!Vr(i.x, i.y) : Gn(i.x, i.y);
    m(G, (kn(i.x, i.y) || o, "grab"), !0);
  }
  function os(i) {
    if (a(Mt)) {
      const f = h.getBoundingClientRect();
      m(je, zt.pan_x + (i.clientX - zt.client_x) * de() / Math.max(1, f.width)), m(Ve, zt.pan_y + (i.clientY - zt.client_y) * we() / Math.max(1, f.height)), m(_, "正在平移画布视图（仅本地）"), pe();
      return;
    }
    const o = Jt(i);
    if (!a(F) && !a(ee)) {
      Wr(o);
      return;
    }
    if (m(G, "grabbing"), a(F))
      ze(Object.assign(Object.assign({}, a(A)), {
        center_x: tt.center_x + o.x - tt.x,
        center_y: tt.center_y + o.y - tt.y
      })), m(_, "正在拖动 " + (Y() ? he() : "版图") + "（本地）");
    else if (a(ee)) {
      const f = Math.atan2(o.y - a(A).center_y, o.x - a(A).center_x) * 180 / Math.PI;
      ze(Object.assign(Object.assign({}, a(A)), {
        rotation_deg: lr(ae.rotation + f - ae.angle)
      })), m(_, "正在旋转 " + (Y() ? he() : "版图") + "（本地）");
    }
    pe();
  }
  function us() {
    !a(F) && !a(ee) && !a(Mt) && m(G, "grab");
  }
  function Zn(i) {
    if (a(Mt)) {
      m(Mt, !1);
      try {
        h.releasePointerCapture(i.pointerId);
      } catch {
      }
      m(_, "画布视图已调整（仅本地，不影响保存数据）"), Wr(Jt(i));
      return;
    }
    if (a(F) || a(ee)) {
      m(F, !1), m(ee, !1);
      try {
        h.releasePointerCapture(i.pointerId);
      } catch {
      }
      Wr(Jt(i));
      const f = (Y() ? he() : "Canvas") + " 已在本地更新，停止操作 250ms 后静默同步";
      Y() ? pr("canvas", f) : bt("canvas", f);
    }
  }
  function Jn(i) {
    return i.deltaMode === WheelEvent.DOM_DELTA_LINE ? i.deltaY * 16 : i.deltaMode === WheelEvent.DOM_DELTA_PAGE ? i.deltaY * Math.max(1, h.getBoundingClientRect().height) : i.deltaY;
  }
  function ls(i) {
    i.preventDefault();
    const o = h.getBoundingClientRect(), f = (i.clientX - o.left) / Math.max(1, o.width) * de(), v = (i.clientY - o.top) / Math.max(1, o.height) * we();
    if (i.ctrlKey || i.metaKey) {
      const J = (f - a(je)) / a(X), W = (v - a(Ve)) / a(X), Z = Te(a(X) * Math.exp(-Jn(i) * 15e-4), 0.2, 8);
      m(je, f - J * Z), m(Ve, v - W * Z), m(X, Z, !0), m(_, "画布视图 " + Math.round(a(X) * 100) + "%（仅本地）"), pe();
      return;
    }
    if (!be()) return;
    const g = Jt(i);
    if (Y()) {
      const J = Vr(g.x, g.y);
      J && $t(J);
    }
    const x = kr(g.x, g.y, a(A)), w = i.shiftKey ? 0.1 : i.altKey ? 5 : 1, N = Math.exp(-Jn(i) * s * w), P = Te(Number(a(A).scale || 1) * N, 0.01, 20);
    let L = Object.assign(Object.assign({}, a(A)), { scale: P });
    const R = Zt(x.x, x.y, L);
    L = Object.assign(Object.assign({}, L), {
      center_x: Number(L.center_x || 0) + g.x - R.x,
      center_y: Number(L.center_y || 0) + g.y - R.y
    }), ze(L), pr("canvas", "本地缩放 " + (P * 100).toFixed(2) + "%；停止 250ms 后同步"), pe();
  }
  function fs() {
    be() && (ze(Object.assign(Object.assign({}, a(A)), {
      center_x: de() / 2,
      center_y: we() / 2,
      scale: 1,
      rotation_deg: 0
    })), bt("reset", "居中归一 " + (Y() ? he() : "版图") + "：scale=1，rotation=0"));
  }
  function cs() {
    be() && (ze(Object.assign(Object.assign({}, a(A)), { center_x: de() / 2, center_y: we() / 2 })), bt("center", "居中 " + (Y() ? he() : "版图") + "：保留缩放和旋转"));
  }
  function hs() {
    if (!be()) return;
    const [i, o, f, v] = Pn(), g = Math.max(1, f - i + 1), x = Math.max(1, v - o + 1), w = Te(Math.min(de() / g, we() / x) * 0.9, 0.01, 20);
    ze(Object.assign(Object.assign({}, a(A)), {
      center_x: de() / 2,
      center_y: we() / 2,
      scale: w
    })), bt("fit", "适配 " + (Y() ? he() : "版图") + "：scale=" + w.toFixed(3));
  }
  function ds() {
    if (!be()) return;
    const i = cr(), o = Math.min(...i.map((P) => P.x)), f = Math.max(...i.map((P) => P.x)), v = Math.min(...i.map((P) => P.y)), g = Math.max(...i.map((P) => P.y));
    let x = 0, w = 0;
    const N = Math.max(20, de() * 0.03);
    if (f < N ? x = N - f : o > de() - N && (x = de() - N - o), g < N ? w = N - g : v > we() - N && (w = we() - N - v), x === 0 && w === 0) {
      m(_, (Y() ? he() : "版图") + " 已经在视野内");
      return;
    }
    ze(Object.assign(Object.assign({}, a(A)), {
      center_x: Number(a(A).center_x || 0) + x,
      center_y: Number(a(A).center_y || 0) + w
    })), bt("bring_into_view", "已找回 " + (Y() ? he() : "版图") + " 到视野内");
  }
  j("keydown", hi, Ka), j("keyup", hi, Qa);
  {
    let i = xt(() => a(F) || a(ee) || a(Mt) ? "focus" : "base");
    Ul(e, {
      get visible() {
        return d.shared.visible;
      },
      variant: "solid",
      get border_mode() {
        return a(i);
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
      children: (o, f) => {
        var v = Kl(), g = $(v);
        {
          var x = (se) => {
            var me = Wl(), oe = z($(me), 2);
            yr(oe, 21, Xe, _r, (ft, ct) => {
              var Ue = kl(), yt = $(Ue), ie = {};
              et(
                (Kt, Pe) => {
                  _e(yt, Kt), ie !== (ie = Pe) && (Ue.value = (Ue.__value = Pe) ?? "");
                },
                [
                  () => a(ct).label || "Label " + String(a(ct).group_id),
                  () => k(a(ct).group_id)
                ]
              ), le(ft, Ue);
            });
            var ue;
            ea(oe);
            var Ae = z(oe, 2), Ne = $(Ae);
            let ut;
            var Me = z(Ne, 2);
            let lt;
            var mt = z(Ae, 2);
            {
              var Qe = (ft) => {
                var ct = Xl(), Ue = z($(ct), 2);
                yr(Ue, 21, Xe, _r, (yt, ie, Kt) => {
                  var Pe = Vl();
                  let ti;
                  var gr = $(Pe);
                  nr(gr, "draggable", !0);
                  var ri = z(gr, 2);
                  ri.textContent = Kt + 1;
                  var ni = z(ri, 2), ii = $(ni), Ss = z(ii);
                  {
                    var Ts = (re) => {
                      var Yr = jl();
                      le(re, Yr);
                    }, ws = xt(() => k(a(ie).group_id) === a(T));
                    Ye(Ss, (re) => {
                      a(ws) && re(Ts);
                    });
                  }
                  var br = z(ni, 2), As = $(br), ai = z(br, 2);
                  let si;
                  et(
                    (re, Yr, Ms, Os, Hs) => {
                      ti = Et(Pe, 1, "group-reorder-item svelte-r41nsf", null, ti, re), _e(ii, `${Yr ?? ""} `), nr(br, "title", Ms), _e(As, Os), si = Et(ai, 1, "solo-button svelte-r41nsf", null, si, Hs);
                    },
                    [
                      () => ({
                        active: k(a(ie).group_id) === a(T),
                        "drop-before": a(Re) === k(a(ie).group_id) && a(He) === "before",
                        "drop-after": a(Re) === k(a(ie).group_id) && a(He) === "after"
                      }),
                      () => a(ie).label || "Label " + String(a(ie).group_id),
                      () => fr(k(a(ie).group_id)) ? "隐藏此层" : "显示此层",
                      () => fr(k(a(ie).group_id)) ? "👁" : "—",
                      () => ({
                        active: a(De) === k(a(ie).group_id)
                      })
                    ]
                  ), j("dragstart", gr, (re) => es(re, k(a(ie).group_id))), j("dragend", gr, ns), j("click", br, (re) => Wa(re, k(a(ie).group_id))), j("click", ai, (re) => za(re, k(a(ie).group_id))), j("click", Pe, (re) => Yn(re, k(a(ie).group_id))), j("keydown", Pe, (re) => {
                    (re.key === "Enter" || re.key === " ") && (re.preventDefault(), Yn(re, k(a(ie).group_id)));
                  }), j("dragover", Pe, (re) => ts(re, k(a(ie).group_id))), j("drop", Pe, (re) => rs(re, k(a(ie).group_id))), le(yt, Pe);
                }), le(ft, ct);
              }, _t = xt(() => Ot());
              Ye(mt, (ft) => {
                a(_t) && ft(Qe);
              });
            }
            et(() => {
              ue !== (ue = a(T)) && (oe.value = (oe.__value = a(T)) ?? "", Or(oe, a(T))), ut = Et(Ne, 1, "svelte-r41nsf", null, ut, { active: a(it) === "align" }), lt = Et(Me, 1, "svelte-r41nsf", null, lt, { active: a(it) === "overlap" });
            }), j("change", oe, as), j("click", Ne, () => Xn("align")), j("click", Me, () => Xn("overlap")), le(se, me);
          }, w = xt(() => Y());
          Ye(g, (se) => {
            a(w) && se(x);
          });
        }
        var N = z(g, 2), P = $(N), L = z(P, 2), R = z(L, 2), J = z(R, 2), W = z(J, 2), Z = z(W, 2), Ie = $(Z), $e = z(Z, 2), Ke = z(N, 2);
        {
          var ps = (se) => {
            var me = Yl();
            yr(
              me,
              20,
              () => [
                ["x", "X", "px"],
                ["y", "Y", "px"],
                ["scale", "缩放", "%"],
                ["rotation", "旋转", "°"],
                ["alpha", "透明度", "%"]
              ],
              _r,
              (oe, ue) => {
                var Ae = zl(), Ne = $(Ae), ut = $(Ne), Me = z(Ne, 2), lt = z(Me, 2), mt = $(lt);
                et(() => {
                  _e(ut, ue[1]), Zo(Me, a(at)[ue[0]]), _e(mt, ue[2]);
                }), j("focus", Me, () => m(ur, ue[0], !0)), j("input", Me, (Qe) => Za(Qe, ue[0])), j("keydown", Me, (Qe) => $a(Qe, ue[0])), j("blur", Me, () => Xr(ue[0])), le(oe, Ae);
              }
            ), le(se, me);
          }, vs = xt(() => be());
          Ye(Ke, (se) => {
            a(vs) && se(ps);
          });
        }
        var $n = z(Ke, 2);
        {
          var gs = (se) => {
            var me = Jl();
            let oe;
            var ue = $(me), Ae = $(ue), Ne = $(Ae), ut = z(Ae, 2), Me = $(ut), lt = z(ue, 2);
            {
              var mt = (_t) => {
                var ft = Zl();
                yr(ft, 21, Cn, _r, (ct, Ue) => {
                  var yt = ql(), ie = $(yt);
                  et(
                    (Kt, Pe) => _e(ie, `应用 ${a(Ue).candidate_id ?? ""}
								· 支持率 ${Kt ?? ""}%
								· 分数 ${Pe ?? ""}`),
                    [
                      () => Math.round(ne(a(Ue).metrics?.central_support, 0) * 100),
                      () => ne(a(Ue).metrics?.score, 0).toFixed(3)
                    ]
                  ), j("click", yt, () => Ln(a(Ue))), le(ct, yt);
                }), le(_t, ft);
              }, Qe = xt(() => Cn().length);
              Ye(lt, (_t) => {
                a(Qe) && _t(mt);
              });
            }
            et(() => {
              oe = Et(me, 1, "alignment-proposal svelte-r41nsf", null, oe, { ambiguous: a(ce).ambiguous === !0 }), _e(Ne, a(ce).ambiguous ? "存在周期歧义" : "粗对齐建议"), _e(Me, a(ce).status);
            }), le(se, me);
          };
          Ye($n, (se) => {
            a(ce) && se(gs);
          });
        }
        var Kn = z($n, 2), ot = $(Kn);
        ia(ot, (se) => h = se, () => h);
        var Qn = z(Kn, 2);
        {
          var bs = (se) => {
            var me = $l(), oe = $(me), ue = $(oe), Ae = z(oe, 2), Ne = $(Ae), ut = z(Ae, 2), Me = $(ut);
            et(
              (lt, mt, Qe, _t) => {
                _e(ue, `当前：${lt ?? ""}`), _e(Ne, `下一层：${mt ?? ""}`), _e(Me, `${Qe ?? ""} × ${_t ?? ""}`);
              },
              [
                () => he(),
                () => I.get(qt())?.view.label,
                () => he(),
                () => I.get(qt())?.view.label
              ]
            ), le(se, me);
          }, ms = xt(() => Y() && a(it) === "overlap" && qt());
          Ye(Qn, (se) => {
            a(ms) && se(bs);
          });
        }
        var _s = z(Qn, 2), zr = $(_s);
        let ei;
        var ys = $(zr), xs = z(zr, 2), Es = $(xs);
        et(
          (se, me, oe, ue, Ae, Ne) => {
            Mr(v, se), P.disabled = me, L.disabled = oe, R.disabled = ue, J.disabled = Ae, _e(Ie, `${Ne ?? ""}%`), Mr(ot, "cursor:" + a(G)), ei = Et(zr, 1, "svelte-r41nsf", null, ei, { dirty: Le > te }), _e(ys, Le > te ? ge === null ? "本地修改" : "同步中" : "已同步"), _e(Es, a(_));
          },
          [
            () => "min-height:" + Ma(d.props.height),
            () => !be(),
            () => !be(),
            () => !be(),
            () => !be(),
            () => Math.round(a(X) * 100)
          ]
        ), j("click", P, fs), j("click", L, cs), j("click", R, hs), j("click", J, ds), j("click", W, () => Wn(0.8)), j("click", Z, Ya), j("click", $e, () => Wn(1.25)), j("pointerdown", ot, ss), j("pointermove", ot, os), j("pointerup", ot, Zn), j("pointercancel", ot, Zn), j("pointerleave", ot, us), j("wheel", ot, ls), le(o, v);
      },
      $$slots: { default: !0 }
    });
  }
  qi();
}
export {
  ef as default
};

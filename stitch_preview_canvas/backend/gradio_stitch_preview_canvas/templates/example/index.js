import { e as x, c as U, a as S, b as I, g as Y, i as j, T as q, r as y, d as T, p as G, f as M, h as z, m as J, s as K, j as Q, E as W, C as O, k as V, u as X, l as D, n as L, o as A, q as Z, t as _, v as P, w as $, x as ee, y as te, P as re, z as se, A as ne, D as ae, B as ie, F as fe, G as le, H as ce, I as ue, S as oe, L as de, J as he, K as _e, M as ve, N as ge, O as me, Q as N } from "./render-CYnyqGu-.js";
x();
let p = !1;
function pe(s) {
  var e = p;
  try {
    return p = !1, [s(), p];
  } finally {
    p = e;
  }
}
const be = (
  // We gotta write it like this because after downleveling the pure comment may end up in the wrong location
  globalThis?.window?.trustedTypes && /* @__PURE__ */ globalThis.window.trustedTypes.createPolicy("svelte-trusted-html", {
    /** @param {string} html */
    createHTML: (s) => s
  })
);
function Te(s) {
  return (
    /** @type {string} */
    be?.createHTML(s) ?? s
  );
}
function Ee(s) {
  var e = U("template");
  return e.innerHTML = Te(s.replaceAll("<!>", "<!---->")), e.content;
}
function k(s, e) {
  var t = (
    /** @type {Effect} */
    I
  );
  t.nodes === null && (t.nodes = { start: s, end: e, a: null, t: null });
}
// @__NO_SIDE_EFFECTS__
function Se(s, e) {
  var t = (e & q) !== 0, r, a = !s.startsWith("<!>");
  return () => {
    r === void 0 && (r = Ee(a ? s : "<!>" + s), r = /** @type {TemplateNode} */
    Y(r));
    var n = (
      /** @type {TemplateNode} */
      t || j ? document.importNode(r, !0) : r.cloneNode(!0)
    );
    return k(n, n), n;
  };
}
function Ae() {
  var s = document.createDocumentFragment(), e = document.createComment(""), t = S();
  return s.append(e, t), k(e, t), s;
}
function R(s, e) {
  s !== null && s.before(
    /** @type {Node} */
    e
  );
}
class Pe {
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
  #s = /* @__PURE__ */ new Set();
  /**
   * Whether to pause (i.e. outro) on change, or destroy immediately.
   * This is necessary for `<svelte:element>`
   */
  #n = !0;
  /**
   * @param {TemplateNode} anchor
   * @param {boolean} transition
   */
  constructor(e, t = !0) {
    this.anchor = e, this.#n = t;
  }
  /**
   * @param {Batch} batch
   */
  #a = (e) => {
    if (this.#t.has(e)) {
      var t = (
        /** @type {Key} */
        this.#t.get(e)
      ), r = this.#r.get(t);
      if (r)
        y(r), this.#s.delete(t);
      else {
        var a = this.#e.get(t);
        a && (y(a.effect), this.#r.set(t, a.effect), this.#e.delete(t), a.fragment.lastChild.remove(), this.anchor.before(a.fragment), r = a.effect);
      }
      for (const [n, i] of this.#t) {
        if (this.#t.delete(n), n === e)
          break;
        const f = this.#e.get(i);
        f && (T(f.effect), this.#e.delete(i));
      }
      for (const [n, i] of this.#r) {
        if (n === t || this.#s.has(n)) continue;
        const f = () => {
          if (Array.from(this.#t.values()).includes(n)) {
            var c = document.createDocumentFragment();
            J(i, c), c.append(S()), this.#e.set(n, { effect: i, fragment: c });
          } else
            T(i);
          this.#s.delete(n), this.#r.delete(n);
        };
        this.#n || !r ? (this.#s.add(n), G(i, f, !1)) : f();
      }
    }
  };
  /**
   * @param {Batch} batch
   */
  #i = (e) => {
    this.#t.delete(e);
    const t = Array.from(this.#t.values());
    for (const [r, a] of this.#e)
      t.includes(r) || (T(a.effect), this.#e.delete(r));
  };
  /**
   *
   * @param {any} key
   * @param {null | ((target: TemplateNode) => void)} fn
   */
  ensure(e, t) {
    var r = (
      /** @type {Batch} */
      z
    ), a = K();
    if (t && !this.#r.has(e) && !this.#e.has(e))
      if (a) {
        var n = document.createDocumentFragment(), i = S();
        n.append(i), this.#e.set(e, {
          effect: M(() => t(i)),
          fragment: n
        });
      } else
        this.#r.set(
          e,
          M(() => t(this.anchor))
        );
    if (this.#t.set(r, e), a) {
      for (const [f, l] of this.#r)
        f === e ? r.unskip_effect(l) : r.skip_effect(l);
      for (const [f, l] of this.#e)
        f === e ? r.unskip_effect(l.effect) : r.skip_effect(l.effect);
      r.oncommit(this.#a), r.ondiscard(this.#i);
    } else
      this.#a(r);
  }
}
function we(s, e, t = !1) {
  var r = new Pe(s), a = t ? W : 0;
  function n(i, f) {
    r.ensure(i, f);
  }
  Q(() => {
    var i = !1;
    e((f, l = 0) => {
      i = !0, n(l, f);
    }), i || n(-1, null);
  }, a);
}
const C = [...` 	
\r\f \v\uFEFF`];
function ye(s, e, t) {
  var r = "" + s;
  if (t) {
    for (var a of Object.keys(t))
      if (t[a])
        r = r ? r + " " + a : a;
      else if (r.length)
        for (var n = a.length, i = 0; (i = r.indexOf(a, i)) >= 0; ) {
          var f = i + n;
          (i === 0 || C.includes(r[i - 1])) && (f === r.length || C.includes(r[f])) ? r = (i === 0 ? "" : r.substring(0, i)) + r.substring(f + 1) : i = f;
        }
  }
  return r === "" ? null : r;
}
function Me(s, e, t, r, a, n) {
  var i = (
    /** @type {any} */
    s[O]
  );
  if (i !== t || i === void 0) {
    var f = ye(t, r, n);
    f == null ? s.removeAttribute("class") : s.className = f, s[O] = t;
  } else if (n && a !== n)
    for (var l in n) {
      var c = !!n[l];
      (a == null || c !== !!a[l]) && s.classList.toggle(l, c);
    }
  return n;
}
function Oe(s = !1) {
  const e = (
    /** @type {ComponentContextLegacy} */
    V
  ), t = e.l.u;
  if (!t) return;
  let r = () => P(e.s);
  if (s) {
    let a = 0, n = (
      /** @type {Record<string, any>} */
      {}
    );
    const i = $(() => {
      let f = !1;
      const l = e.s;
      for (const c in l)
        l[c] !== n[c] && (n[c] = l[c], f = !0);
      return f && a++, a;
    });
    r = () => _(i);
  }
  t.b.length && X(() => {
    F(e, r), L(t.b);
  }), D(() => {
    const a = A(() => t.m.map(Z));
    return () => {
      for (const n of a)
        typeof n == "function" && n();
    };
  }), t.a.length && D(() => {
    F(e, r), L(t.a);
  });
}
function F(s, e) {
  if (s.l.s)
    for (const t of s.l.s) _(t);
  e();
}
function E(s, e, t, r) {
  var a = !ie || (t & fe) !== 0, n = (t & le) !== 0, i = (
    /** @type {V} */
    r
  ), f = !0, l = () => (f && (f = !1, i = /** @type {V} */
  r), i);
  let c;
  {
    var v = oe in s || de in s;
    c = ee(s, e)?.set ?? (v && e in s ? (u) => s[e] = u : void 0);
  }
  var o, g = !1;
  [o, g] = pe(() => (
    /** @type {V} */
    s[e]
  )), o === void 0 && r !== void 0 && (o = l(), c && (a && te(), c(o)));
  var d;
  if (a ? d = () => {
    var u = (
      /** @type {V} */
      s[e]
    );
    return u === void 0 ? l() : (f = !0, u);
  } : d = () => {
    var u = (
      /** @type {V} */
      s[e]
    );
    return u !== void 0 && (i = /** @type {V} */
    void 0), u === void 0 ? i : u;
  }, a && (t & re) === 0)
    return d;
  if (c) {
    var B = s.$$legacy;
    return (
      /** @type {() => V} */
      (function(u, m) {
        return arguments.length > 0 ? ((!a || !m || B || g) && c(m ? d() : u), u) : d();
      })
    );
  }
  var b = !1, h = ce(() => (b = !1, d()));
  _(h);
  var H = (
    /** @type {Effect} */
    I
  );
  return (
    /** @type {() => V} */
    (function(u, m) {
      if (arguments.length > 0) {
        const w = m ? _(h) : a && n ? se(u) : u;
        return ne(h, w), b = !0, i !== void 0 && (i = w), u;
      }
      return ue && b || (H.f & ae) !== 0 ? h.v : _(h);
    })
  );
}
var De = /* @__PURE__ */ Se('<div><span class="label"> </span></div>');
function Ne(s, e) {
  ve(e, !1);
  let t = E(e, "value", 8), r = E(e, "type", 8), a = E(e, "selected", 8, !1);
  Oe();
  var n = Ae(), i = he(n);
  {
    var f = (l) => {
      var c = De();
      let v;
      var o = N(c), g = N(o);
      ge(() => {
        v = Me(c, 1, "container svelte-s3apn9", null, v, {
          table: r() === "table",
          gallery: r() === "gallery",
          selected: a()
        }), me(g, `tile ${P(t()), A(() => t().selected ?? 0) ?? ""}`);
      }), R(l, c);
    };
    we(i, (l) => {
      P(t()), A(() => t() && t().tiles && t().tiles.length > 0) && l(f);
    });
  }
  R(s, n), _e();
}
export {
  Ne as default
};

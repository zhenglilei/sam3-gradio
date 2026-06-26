import { W as z, w as A, X as U, Y as I, t as W, Z as k, _ as x, b, $ as C, j as Y, f as _, a as L, p as N, h as p, m as Q, a0 as X, a1 as T, a2 as F, a3 as B, a4 as Z, a5 as G, a6 as H, d as O, a7 as w, a8 as q, n as $, E as J, a9 as K, aa as ee, ab as te, ac as re, ad as ie, ae as se, af as ne, R as ae, Q as fe } from "./runtime-CpL30OSY.js";
function he(r) {
  let e = 0, t = I(0), s;
  return () => {
    z() && (A(t), U(() => (e === 0 && (s = W(() => r(() => x(t)))), e += 1, () => {
      k(() => {
        e -= 1, e === 0 && (s?.(), s = void 0, x(t));
      });
    })));
  };
}
var oe = J | K;
function le(r, e, t, s) {
  new ce(r, e, t, s);
}
class ce {
  /** @type {Boundary | null} */
  parent;
  is_pending = !1;
  /**
   * API-level transformError transform function. Transforms errors before they reach the `failed` snippet.
   * Inherited from parent boundary, or defaults to identity.
   * @type {(error: unknown) => unknown}
   */
  transform_error;
  /** @type {TemplateNode} */
  #i;
  /** @type {TemplateNode | null} */
  #b = null;
  /** @type {BoundaryProps} */
  #s;
  /** @type {((anchor: Node) => void)} */
  #o;
  /** @type {Effect} */
  #t;
  /** @type {Effect | null} */
  #n = null;
  /** @type {Effect | null} */
  #e = null;
  /** @type {Effect | null} */
  #r = null;
  /** @type {DocumentFragment | null} */
  #a = null;
  #l = 0;
  #h = 0;
  #c = !1;
  /** @type {Set<Effect>} */
  #_ = /* @__PURE__ */ new Set();
  /** @type {Set<Effect>} */
  #p = /* @__PURE__ */ new Set();
  /**
   * A source containing the number of pending async deriveds/expressions.
   * Only created if `$effect.pending()` is used inside the boundary,
   * otherwise updating the source results in needless `Batch.ensure()`
   * calls followed by no-op flushes
   * @type {Source<number> | null}
   */
  #f = null;
  #y = he(() => (this.#f = I(this.#l), () => {
    this.#f = null;
  }));
  /**
   * @param {TemplateNode} node
   * @param {BoundaryProps} props
   * @param {((anchor: Node) => void)} children
   * @param {((error: unknown) => unknown) | undefined} [transform_error]
   */
  constructor(e, t, s, a) {
    this.#i = e, this.#s = t, this.#o = (i) => {
      var l = (
        /** @type {Effect} */
        b
      );
      l.b = this, l.f |= C, s(i);
    }, this.parent = /** @type {Effect} */
    b.b, this.transform_error = a ?? this.parent?.transform_error ?? ((i) => i), this.#t = Y(() => {
      this.#v();
    }, oe);
  }
  #w() {
    try {
      this.#n = _(() => this.#o(this.#i));
    } catch (e) {
      this.error(e);
    }
  }
  /**
   * @param {unknown} error The deserialized error from the server's hydration comment
   */
  #E(e) {
    const t = this.#s.failed;
    t && (this.#r = _(() => {
      t(
        this.#i,
        () => e,
        () => () => {
        }
      );
    }));
  }
  #S() {
    const e = this.#s.pending;
    e && (this.is_pending = !0, this.#e = _(() => e(this.#i)), k(() => {
      var t = this.#a = document.createDocumentFragment(), s = L();
      t.append(s), this.#n = this.#d(() => _(() => this.#o(s))), this.#h === 0 && (this.#i.before(t), this.#a = null, N(
        /** @type {Effect} */
        this.#e,
        () => {
          this.#e = null;
        }
      ), this.#u(
        /** @type {Batch} */
        p
      ));
    }));
  }
  #v() {
    try {
      if (this.is_pending = this.has_pending_snippet(), this.#h = 0, this.#l = 0, this.#n = _(() => {
        this.#o(this.#i);
      }), this.#h > 0) {
        var e = this.#a = document.createDocumentFragment();
        Q(this.#n, e);
        const t = (
          /** @type {(anchor: Node) => void} */
          this.#s.pending
        );
        this.#e = _(() => t(this.#i));
      } else
        this.#u(
          /** @type {Batch} */
          p
        );
    } catch (t) {
      this.error(t);
    }
  }
  /**
   * @param {Batch} batch
   */
  #u(e) {
    this.is_pending = !1, e.transfer_effects(this.#_, this.#p);
  }
  /**
   * Defer an effect inside a pending boundary until the boundary resolves
   * @param {Effect} effect
   */
  defer_effect(e) {
    X(e, this.#_, this.#p);
  }
  /**
   * Returns `false` if the effect exists inside a boundary whose pending snippet is shown
   * @returns {boolean}
   */
  is_rendered() {
    return !this.is_pending && (!this.parent || this.parent.is_rendered());
  }
  has_pending_snippet() {
    return !!this.#s.pending;
  }
  /**
   * @template T
   * @param {() => T} fn
   */
  #d(e) {
    var t = b, s = q, a = $;
    T(this.#t), F(this.#t), B(this.#t.ctx);
    try {
      return Z.ensure(), e();
    } catch (i) {
      return G(i), null;
    } finally {
      T(t), F(s), B(a);
    }
  }
  /**
   * Updates the pending count associated with the currently visible pending snippet,
   * if any, such that we can replace the snippet with content once work is done
   * @param {1 | -1} d
   * @param {Batch} batch
   */
  #g(e, t) {
    if (!this.has_pending_snippet()) {
      this.parent && this.parent.#g(e, t);
      return;
    }
    this.#h += e, this.#h === 0 && (this.#u(t), this.#e && N(this.#e, () => {
      this.#e = null;
    }), this.#a && (this.#i.before(this.#a), this.#a = null));
  }
  /**
   * Update the source that powers `$effect.pending()` inside this boundary,
   * and controls when the current `pending` snippet (if any) is removed.
   * Do not call from inside the class
   * @param {1 | -1} d
   * @param {Batch} batch
   */
  update_pending_count(e, t) {
    this.#g(e, t), this.#l += e, !(!this.#f || this.#c) && (this.#c = !0, k(() => {
      this.#c = !1, this.#f && H(this.#f, this.#l);
    }));
  }
  get_effect_pending() {
    return this.#y(), A(
      /** @type {Source<number>} */
      this.#f
    );
  }
  /** @param {unknown} error */
  error(e) {
    if (!this.#s.onerror && !this.#s.failed)
      throw e;
    p?.is_fork ? (this.#n && p.skip_effect(this.#n), this.#e && p.skip_effect(this.#e), this.#r && p.skip_effect(this.#r), p.oncommit(() => {
      this.#m(e);
    })) : this.#m(e);
  }
  /**
   * @param {unknown} error
   */
  #m(e) {
    this.#n && (O(this.#n), this.#n = null), this.#e && (O(this.#e), this.#e = null), this.#r && (O(this.#r), this.#r = null);
    var t = this.#s.onerror;
    let s = this.#s.failed;
    var a = !1, i = !1;
    const l = () => {
      if (a) {
        te();
        return;
      }
      a = !0, i && ee(), this.#r !== null && N(this.#r, () => {
        this.#r = null;
      }), this.#d(() => {
        this.#v();
      });
    }, d = (n) => {
      try {
        i = !0, t?.(n, l), i = !1;
      } catch (f) {
        w(f, this.#t && this.#t.parent);
      }
      s && (this.#r = this.#d(() => {
        try {
          return _(() => {
            var f = (
              /** @type {Effect} */
              b
            );
            f.b = this, f.f |= C, s(
              this.#i,
              () => n,
              () => l
            );
          });
        } catch (f) {
          return w(
            f,
            /** @type {Effect} */
            this.#t.parent
          ), null;
        }
      }));
    };
    k(() => {
      var n;
      try {
        n = this.transform_error(e);
      } catch (f) {
        w(f, this.#t && this.#t.parent);
        return;
      }
      n !== null && typeof n == "object" && typeof /** @type {any} */
      n.then == "function" ? n.then(
        d,
        /** @param {unknown} e */
        (f) => w(f, this.#t && this.#t.parent)
      ) : d(n);
    });
  }
}
const E = /* @__PURE__ */ Symbol("events"), ue = /* @__PURE__ */ new Set(), D = /* @__PURE__ */ new Set();
let M = null;
function j(r) {
  var e = this, t = (
    /** @type {Node} */
    e.ownerDocument
  ), s = r.type, a = r.composedPath?.() || [], i = (
    /** @type {null | Element} */
    a[0] || r.target
  );
  M = r;
  var l = 0, d = M === r && r[E];
  if (d) {
    var n = a.indexOf(d);
    if (n !== -1 && (e === document || e === /** @type {any} */
    window)) {
      r[E] = e;
      return;
    }
    var f = a.indexOf(e);
    if (f === -1)
      return;
    n <= f && (l = n);
  }
  if (i = /** @type {Element} */
  a[l] || r.target, i !== e) {
    re(r, "currentTarget", {
      configurable: !0,
      get() {
        return i || t;
      }
    });
    var v = q, m = b;
    F(null), T(null);
    try {
      for (var u, c = []; i !== null && i !== e; ) {
        try {
          var o = i[E]?.[s];
          o != null && (!/** @type {any} */
          i.disabled || // DOM could've been updated already by the time this is reached, so we check this as well
          // -> the target could not have been disabled because it emits the event in the first place
          r.target === i) && o.call(i, r);
        } catch (h) {
          u ? c.push(h) : u = h;
        }
        if (r.cancelBubble) break;
        l++, i = l < a.length ? (
          /** @type {Element} */
          a[l]
        ) : null;
      }
      if (u) {
        for (let h of c)
          queueMicrotask(() => {
            throw h;
          });
        throw u;
      }
    } finally {
      r[E] = e, delete r.currentTarget, F(v), T(m);
    }
  }
}
const de = ["touchstart", "touchmove"];
function _e(r) {
  return de.includes(r);
}
function me(r, e) {
  return pe(r, e);
}
const S = /* @__PURE__ */ new Map();
function pe(r, { target: e, anchor: t, props: s = {}, events: a, context: i, intro: l = !0, transformError: d }) {
  ie();
  var n = void 0, f = se(() => {
    var v = t ?? e.appendChild(L());
    le(
      /** @type {TemplateNode} */
      v,
      {
        pending: () => {
        }
      },
      (c) => {
        ae({});
        var o = (
          /** @type {ComponentContext} */
          $
        );
        i && (o.c = i), a && (s.$$events = a), n = r(c, s) || {}, fe();
      },
      d
    );
    var m = /* @__PURE__ */ new Set(), u = (c) => {
      for (var o = 0; o < c.length; o++) {
        var h = c[o];
        if (!m.has(h)) {
          m.add(h);
          var y = _e(h);
          for (const R of [e, document]) {
            var g = S.get(R);
            g === void 0 && (g = /* @__PURE__ */ new Map(), S.set(R, g));
            var V = g.get(h);
            V === void 0 ? (R.addEventListener(h, j, { passive: y }), g.set(h, 1)) : g.set(h, V + 1);
          }
        }
      }
    };
    return u(ne(ue)), D.add(u), () => {
      for (var c of m)
        for (const y of [e, document]) {
          var o = (
            /** @type {Map<string, number>} */
            S.get(y)
          ), h = (
            /** @type {number} */
            o.get(c)
          );
          --h == 0 ? (y.removeEventListener(c, j), o.delete(c), o.size === 0 && S.delete(y)) : o.set(c, h);
        }
      D.delete(u), v !== t && v.parentNode?.removeChild(v);
    };
  });
  return P.set(n, f), n;
}
let P = /* @__PURE__ */ new WeakMap();
function ye(r, e) {
  const t = P.get(r);
  return t ? (P.delete(r), t(e)) : Promise.resolve();
}
const ve = "5";
typeof window < "u" && ((window.__svelte ??= {}).v ??= /* @__PURE__ */ new Set()).add(ve);
const be = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null
}, Symbol.toStringTag, { value: "Module" }));
export {
  be as SVELTE_VERSION,
  me as mount,
  ye as unmount
};

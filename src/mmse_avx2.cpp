#include "internal/mmse_internal.h"

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <immintrin.h>
#endif

namespace mmse::detail {

namespace {

inline __m256 load8(const float* ptr) {
    return _mm256_loadu_ps(ptr);
}

inline void store8(float* ptr, __m256 v) {
    _mm256_storeu_ps(ptr, v);
}

inline __m256 cmul_re(__m256 ar, __m256 ai, __m256 br, __m256 bi) {
    return _mm256_fnmadd_ps(ai, bi, _mm256_mul_ps(ar, br));
}

inline __m256 cmul_im(__m256 ar, __m256 ai, __m256 br, __m256 bi) {
    return _mm256_fmadd_ps(ar, bi, _mm256_mul_ps(ai, br));
}

inline __m256 cnorm2_v(__m256 re, __m256 im) {
    return _mm256_fmadd_ps(im, im, _mm256_mul_ps(re, re));
}

inline __m256 clamp_v(__m256 v, float lo, float hi) {
    const __m256 vlo = _mm256_set1_ps(lo);
    const __m256 vhi = _mm256_set1_ps(hi);
    return _mm256_min_ps(_mm256_max_ps(v, vlo), vhi);
}

}  // namespace

void equalize_2x2_avx2(const PackedEqualizerInputs& packed,
                       std::uint32_t begin,
                       std::uint32_t end,
                       float sigma2,
                       float det_floor,
                       float g_min,
                       float gamma_max,
                       float* out_re_layer0,
                       float* out_im_layer0,
                       float* out_sinr_layer0,
                       float* out_re_layer1,
                       float* out_im_layer1,
                       float* out_sinr_layer1) {
    const __m256 vsigma2 = _mm256_set1_ps(sigma2);
    const __m256 vdet_floor = _mm256_set1_ps(det_floor);
    const __m256 vgmin = _mm256_set1_ps(g_min);
    const __m256 vone = _mm256_set1_ps(1.0F);
    const __m256 vtwo = _mm256_set1_ps(2.0F);
    const __m256 vgamma_max = _mm256_set1_ps(gamma_max);

    std::uint32_t i = begin;
    for (; i + 8U <= end; i += 8U) {
        const __m256 h00r = load8(&packed.h00_re[i]);
        const __m256 h00i = load8(&packed.h00_im[i]);
        const __m256 h01r = load8(&packed.h01_re[i]);
        const __m256 h01i = load8(&packed.h01_im[i]);
        const __m256 h10r = load8(&packed.h10_re[i]);
        const __m256 h10i = load8(&packed.h10_im[i]);
        const __m256 h11r = load8(&packed.h11_re[i]);
        const __m256 h11i = load8(&packed.h11_im[i]);
        const __m256 y0r = load8(&packed.y0_re[i]);
        const __m256 y0i = load8(&packed.y0_im[i]);
        const __m256 y1r = load8(&packed.y1_re[i]);
        const __m256 y1i = load8(&packed.y1_im[i]);

        const __m256 a11 = _mm256_add_ps(_mm256_add_ps(cnorm2_v(h00r, h00i), cnorm2_v(h10r, h10i)), vsigma2);
        const __m256 a22 = _mm256_add_ps(_mm256_add_ps(cnorm2_v(h01r, h01i), cnorm2_v(h11r, h11i)), vsigma2);
        const __m256 a12r = _mm256_add_ps(cmul_re(h00r, _mm256_sub_ps(_mm256_setzero_ps(), h00i), h01r, h01i),
                                          cmul_re(h10r, _mm256_sub_ps(_mm256_setzero_ps(), h10i), h11r, h11i));
        const __m256 a12i = _mm256_add_ps(cmul_im(h00r, _mm256_sub_ps(_mm256_setzero_ps(), h00i), h01r, h01i),
                                          cmul_im(h10r, _mm256_sub_ps(_mm256_setzero_ps(), h10i), h11r, h11i));
        const __m256 det = _mm256_max_ps(
            _mm256_fnmadd_ps(cnorm2_v(a12r, a12i), vone, _mm256_mul_ps(a11, a22)), vdet_floor);
        const __m256 rcp0 = _mm256_rcp_ps(det);
        const __m256 inv_det = _mm256_mul_ps(rcp0, _mm256_fnmadd_ps(det, rcp0, vtwo));

        const __m256 inv11 = _mm256_mul_ps(a22, inv_det);
        const __m256 inv22 = _mm256_mul_ps(a11, inv_det);
        const __m256 inv12r = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), a12r), inv_det);
        const __m256 inv12i = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), a12i), inv_det);
        const __m256 inv21r = inv12r;
        const __m256 inv21i = _mm256_sub_ps(_mm256_setzero_ps(), inv12i);

        const __m256 hh00r = h00r;
        const __m256 hh00i = _mm256_sub_ps(_mm256_setzero_ps(), h00i);
        const __m256 hh01r = h10r;
        const __m256 hh01i = _mm256_sub_ps(_mm256_setzero_ps(), h10i);
        const __m256 hh10r = h01r;
        const __m256 hh10i = _mm256_sub_ps(_mm256_setzero_ps(), h01i);
        const __m256 hh11r = h11r;
        const __m256 hh11i = _mm256_sub_ps(_mm256_setzero_ps(), h11i);

        const __m256 w00r =
            _mm256_add_ps(_mm256_mul_ps(inv11, hh00r), cmul_re(inv12r, inv12i, hh10r, hh10i));
        const __m256 w00i = _mm256_add_ps(_mm256_mul_ps(inv11, hh00i), cmul_im(inv12r, inv12i, hh10r, hh10i));
        const __m256 w01r =
            _mm256_add_ps(_mm256_mul_ps(inv11, hh01r), cmul_re(inv12r, inv12i, hh11r, hh11i));
        const __m256 w01i = _mm256_add_ps(_mm256_mul_ps(inv11, hh01i), cmul_im(inv12r, inv12i, hh11r, hh11i));
        const __m256 w10r = _mm256_add_ps(cmul_re(inv21r, inv21i, hh00r, hh00i), _mm256_mul_ps(inv22, hh10r));
        const __m256 w10i = _mm256_add_ps(cmul_im(inv21r, inv21i, hh00r, hh00i), _mm256_mul_ps(inv22, hh10i));
        const __m256 w11r = _mm256_add_ps(cmul_re(inv21r, inv21i, hh01r, hh01i), _mm256_mul_ps(inv22, hh11r));
        const __m256 w11i = _mm256_add_ps(cmul_im(inv21r, inv21i, hh01r, hh01i), _mm256_mul_ps(inv22, hh11i));

        const __m256 z0r = _mm256_add_ps(cmul_re(w00r, w00i, y0r, y0i), cmul_re(w01r, w01i, y1r, y1i));
        const __m256 z0i = _mm256_add_ps(cmul_im(w00r, w00i, y0r, y0i), cmul_im(w01r, w01i, y1r, y1i));
        const __m256 z1r = _mm256_add_ps(cmul_re(w10r, w10i, y0r, y0i), cmul_re(w11r, w11i, y1r, y1i));
        const __m256 z1i = _mm256_add_ps(cmul_im(w10r, w10i, y0r, y0i), cmul_im(w11r, w11i, y1r, y1i));

        const __m256 g0 = clamp_v(_mm256_add_ps(cmul_re(w00r, w00i, h00r, h00i), cmul_re(w01r, w01i, h10r, h10i)),
                                  g_min,
                                  1.0F - g_min);
        const __m256 g1 = clamp_v(_mm256_add_ps(cmul_re(w10r, w10i, h01r, h01i), cmul_re(w11r, w11i, h11r, h11i)),
                                  g_min,
                                  1.0F - g_min);

        const __m256 inv_g0 = _mm256_div_ps(vone, g0);
        const __m256 inv_g1 = _mm256_div_ps(vone, g1);
        const __m256 x0r = _mm256_mul_ps(z0r, inv_g0);
        const __m256 x0i = _mm256_mul_ps(z0i, inv_g0);
        const __m256 x1r = _mm256_mul_ps(z1r, inv_g1);
        const __m256 x1i = _mm256_mul_ps(z1i, inv_g1);

        const __m256 gamma0 = _mm256_min_ps(_mm256_div_ps(g0, _mm256_sub_ps(vone, g0)), vgamma_max);
        const __m256 gamma1 = _mm256_min_ps(_mm256_div_ps(g1, _mm256_sub_ps(vone, g1)), vgamma_max);

        store8(out_re_layer0 + i, x0r);
        store8(out_im_layer0 + i, x0i);
        store8(out_sinr_layer0 + i, gamma0);
        store8(out_re_layer1 + i, x1r);
        store8(out_im_layer1 + i, x1i);
        store8(out_sinr_layer1 + i, gamma1);
    }

    for (; i < end; ++i) {
        const Complex32 h00{packed.h00_re[i], packed.h00_im[i]};
        const Complex32 h01{packed.h01_re[i], packed.h01_im[i]};
        const Complex32 h10{packed.h10_re[i], packed.h10_im[i]};
        const Complex32 h11{packed.h11_re[i], packed.h11_im[i]};
        const Complex32 y0{packed.y0_re[i], packed.y0_im[i]};
        const Complex32 y1{packed.y1_re[i], packed.y1_im[i]};
        const EqualizedSymbol eq0 =
            equalize_2x2_scalar(h00, h01, h10, h11, y0, y1, sigma2, det_floor, g_min, gamma_max, 0U);
        const EqualizedSymbol eq1 =
            equalize_2x2_scalar(h00, h01, h10, h11, y0, y1, sigma2, det_floor, g_min, gamma_max, 1U);
        out_re_layer0[i] = eq0.xhat.re;
        out_im_layer0[i] = eq0.xhat.im;
        out_sinr_layer0[i] = eq0.gamma;
        out_re_layer1[i] = eq1.xhat.re;
        out_im_layer1[i] = eq1.xhat.im;
        out_sinr_layer1[i] = eq1.gamma;
    }
}

}  // namespace mmse::detail

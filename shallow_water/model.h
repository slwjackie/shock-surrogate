/* Generated from frozen training snapshots by python -m shallow_water.model. */
#ifndef SW_DEPTH_MODEL_H
#define SW_DEPTH_MODEL_H
#include <math.h>
static const double sw_nn_w[3][3][12] = {{{0x1.7a21746fbb5c4p-1,-0x1.0c33b1be06bbbp-1,-0x1.55fe2692ec104p-1,-0x1.b121bde80f403p-2,0x1.9ee6ed129ba72p-2,0x1.ad7b9d61b6666p-1,0x1.bcde8853c7b4bp-1,-0x1.f06b33088ad2bp-2,0x1.1b03a5afbead0p-1,0x1.2a71a2dfe7843p-1,-0x1.17ef6d87323aap-2,-0x1.74339b38d7a5ep-3},{0x1.bdda419ae49d9p-3,0x1.dcbc1568b0cadp-1,0x1.5c30a7371e364p-2,0x1.3ada83a323727p-9,-0x1.1a0402c0fa686p-1,0x1.8ec40a0ed653cp+0,0x1.1f966b5887a1bp-1,-0x1.923948145fb2cp-2,-0x1.1fea60b0702eap-1,-0x1.19712dbf38d8fp+0,0x1.2feec12ed8293p+0,0x1.18b5aa26adbdfp-1},{-0x1.bbb6c7f5895adp-1,0x1.ff732f925c62bp-2,-0x1.fb05fbde996dcp-1,-0x1.0be53cdf4d577p-7,-0x1.3b641db9b1ad5p-1,-0x1.039107b751fb3p+0,0x1.4efe989f09811p-2,-0x1.7e8f69ad1a17cp+0,0x1.1c14b5e2ebdc1p-1,-0x1.365d8b19d6518p-1,-0x1.499330f08bec1p-2,-0x1.6c8c687e55263p-2}},{{0x1.6a467ea24384cp-1,-0x1.0c33b1be06bbbp-1,-0x1.1faed81c5de50p-1,-0x1.91bb93e96b7d6p-2,0x1.b29ffd17646a9p-2,0x1.ad7b9d61b6666p-1,0x1.b6a0912f758d3p-1,-0x1.ceff9dee8b525p-2,0x1.1c7329f197efep-1,0x1.306b7f53f8182p-1,-0x1.4b61afd7ff5d0p-2,-0x1.74339b38d7a5ep-3},{0x1.f05cdd7a71557p-4,0x1.dcbc1568b0cadp-1,0x1.0841bf8252083p-3,0x1.adb58fd3353f8p-5,-0x1.1e42d9deeba71p-1,0x1.8ec40a0ed653cp+0,0x1.0b66e4264cc20p-1,-0x1.20d3db04fde13p+0,-0x1.3bca64d3c3095p-1,-0x1.223204cce4498p+0,0x1.250499c2b76bap+0,0x1.18b5aa26adbdfp-1},{-0x1.fb617c38941b9p-1,0x1.ff732f925c62bp-2,-0x1.b8a1de6dc8d63p-1,0x1.4ac7ae1e7d7f4p-5,-0x1.5ed34f25d2828p-1,-0x1.039107b751fb3p+0,0x1.6d854f96565a4p-2,-0x1.0d67ae145cd6fp+1,0x1.12d306bce2b5dp-1,-0x1.4b4f1f10770bep-1,-0x1.532cbe1854558p-2,-0x1.6c8c687e55263p-2}},{{0x1.6ae7324708f35p-1,-0x1.0c33b1be06bbbp-1,-0x1.29982340cda84p-1,-0x1.86ef58d4b160dp-2,0x1.a2f161569cbd3p-2,0x1.ad7b9d61b6666p-1,0x1.b385098a10ae8p-1,-0x1.c72431fc7f8bbp-2,0x1.1c79cf4e4ae4ap-1,0x1.2847ec7d880a5p-1,-0x1.2793c60622d90p-2,-0x1.74339b38d7a5ep-3},{0x1.92f663ed01ec7p-3,0x1.dcbc1568b0cadp-1,0x1.015c8874a5f8ep-4,0x1.298d23565d180p-4,-0x1.30deb792bb5dap-1,0x1.8ec40a0ed653cp+0,0x1.00b17f6a21751p-1,-0x1.128041b368a3cp+0,-0x1.3055fad901d64p-1,-0x1.2a26151738e03p+0,0x1.370692ea09bf5p+0,0x1.18b5aa26adbdfp-1},{-0x1.f4805a6b7cfb8p-1,0x1.ff732f925c62bp-2,-0x1.9b2a9db141b4cp-1,0x1.a1a60550dad61p-5,-0x1.505ffb60ee336p-1,-0x1.039107b751fb3p+0,0x1.60b14f389d762p-2,-0x1.fcf16e8af8610p+0,0x1.1996db2d5fd16p-1,-0x1.5613ed0631d57p-1,-0x1.3cff36748ce4dp-2,-0x1.6c8c687e55263p-2}}};
static const double sw_nn_b[3][12] = {{0x1.6d172f1773f5fp-3,-0x1.6651cd8b58eabp-1,0x1.1671209aea02fp-1,0x1.486c45833a5acp-2,0x1.1369ca8bd87f8p-1,-0x1.8e0c6f8c601e8p+0,-0x1.3f77049eaae70p-2,0x1.d5860d416576fp-1,-0x1.5251547b4a3dep-2,0x1.07419812dbd20p+0,0x1.348a6c9ba81c6p-2,-0x1.587bd7b3a6348p-7},{0x1.353fbd9bac80bp-3,-0x1.6651cd8b58eabp-1,0x1.41bb0ab437716p-1,0x1.566c61d03faecp-2,0x1.17e3caa3d5e8ap-1,-0x1.8e0c6f8c601e8p+0,-0x1.57a61192c1c0cp-2,0x1.9e59d4366c397p-1,-0x1.7a731459c16cap-2,0x1.0a816bbc057e6p+0,0x1.05841448adc14p-2,-0x1.587bd7b3a6348p-7},{0x1.0d8de01e65fc2p-3,-0x1.6651cd8b58eabp-1,0x1.49cc825cc9bb3p-1,0x1.54bfa73083bcep-2,0x1.1f666901b1295p-1,-0x1.8e0c6f8c601e8p+0,-0x1.7db86aba9e1e8p-2,0x1.9358f904753fap-1,-0x1.952121ed207a9p-2,0x1.0ad469552a2fdp+0,0x1.fd918df5e9bbdp-3,-0x1.587bd7b3a6348p-7}};
static const double sw_nn_v[3][12] = {{0x1.57f7bff206208p-4,-0x1.a8e5d230edd5dp-12,-0x1.78cb12608cc74p-3,-0x1.74e3913490703p-2,0x1.31a1e84454944p-4,-0x1.f0727da50cef4p-7,0x1.a9aa6a8470682p-4,-0x1.c755dbb5fb2acp-11,0x1.372d97c7b706bp-3,0x1.9d9c6e5321ba5p-4,-0x1.832eee2ef7b42p-10,0x1.02f7c329c44bfp-4},{0x1.47ffeb5aa26d4p-4,-0x1.a8e5d230edd5dp-12,-0x1.8a1af1f65f97fp-3,-0x1.6510116ba80c7p-2,0x1.a281aea068b68p-4,-0x1.f0727da50cef4p-7,0x1.c2a05712afdd9p-4,-0x1.603025d8c08bbp-5,0x1.2dbca7c6421ecp-2,0x1.fc8892ce57eedp-4,0x1.1e065db2ecc95p-8,0x1.02f7c329c44bfp-4},{0x1.40eca61b614a5p-4,-0x1.a8e5d230edd5dp-12,-0x1.b5492852c3415p-3,-0x1.47aef327df54bp-2,0x1.c08adef66d43ap-4,-0x1.f0727da50cef4p-7,0x1.f4c6d0df5f33ep-4,-0x1.98cec0fc9440fp-5,0x1.62a939aad44e7p-2,0x1.0b9e2fa80c8edp-3,-0x1.8c71b30ef7e2bp-8,0x1.02f7c329c44bfp-4}};
static const double sw_nn_c[3] = {0x1.b974ade1d8249p-2,0x1.c702fe56a77e0p-2,0x1.c81cce3a8be1fp-2};
static inline double sw_model_guess(int model, double hmax, double r, double w)
{
    const double base = hmax*w*w;
    if (!(w > r)) return base;
    const double q = (w-r)/w;
    const double p = w > 1.0 ? (w-1.0)/w : 0.0;
    const double t = p/q;
    double correction = sw_nn_c[model];
    for (int i=0; i<12; ++i) {
        const double activation = sw_nn_b[model][i] + q*sw_nn_w[model][0][i]
                              + p*sw_nn_w[model][1][i] + t*sw_nn_w[model][2][i];
        correction += (activation > 0.0 ? activation : 0.0)*sw_nn_v[model][i];
    }
    const double factor = 1.0 - (q*q*q+p*p*p)*correction;
    return base*(factor > 0.0 ? factor : 0.0);
}
#endif

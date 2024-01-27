/*
 * Automatically Generated from Mathematica.
 * Mon 4 Jul 2022 20:56:22 GMT-04:00
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "dJp_left_hip_yaw_src.h"

#ifdef _MSC_VER
  #define INLINE __forceinline /* use __forceinline (VC++ specific) */
#else
  #define INLINE inline        /* use standard inline */
#endif

/**
 * Copied from Wolfram Mathematica C Definitions file mdefs.hpp
 * Changed marcos to inline functions (Eric Cousineau)
 */
INLINE double Power(double x, double y) { return pow(x, y); }
INLINE double Sqrt(double x) { return sqrt(x); }

INLINE double Abs(double x) { return fabs(x); }

INLINE double Exp(double x) { return exp(x); }
INLINE double Log(double x) { return log(x); }

INLINE double Sin(double x) { return sin(x); }
INLINE double Cos(double x) { return cos(x); }
INLINE double Tan(double x) { return tan(x); }

INLINE double Csc(double x) { return 1.0/sin(x); }
INLINE double Sec(double x) { return 1.0/cos(x); }

INLINE double ArcSin(double x) { return asin(x); }
INLINE double ArcCos(double x) { return acos(x); }
//INLINE double ArcTan(double x) { return atan(x); }

/* update ArcTan function to use atan2 instead. */
INLINE double ArcTan(double x, double y) { return atan2(y,x); }

INLINE double Sinh(double x) { return sinh(x); }
INLINE double Cosh(double x) { return cosh(x); }
INLINE double Tanh(double x) { return tanh(x); }

#define E 2.71828182845904523536029
#define Pi 3.14159265358979323846264
#define Degree 0.01745329251994329576924

/*
 * Sub functions
 */
static void output1(double *p_output1,const double *var1,const double *var2)
{
  double t479;
  double t259;
  double t316;
  double t370;
  double t482;
  double t256;
  double t564;
  double t565;
  double t566;
  double t412;
  double t494;
  double t523;
  double t580;
  double t668;
  double t679;
  double t737;
  double t778;
  double t795;
  double t1019;
  double t1156;
  double t1214;
  double t1308;
  double t2232;
  double t3691;
  double t3702;
  double t3716;
  double t1102;
  double t1116;
  double t1811;
  double t1866;
  double t3798;
  double t3821;
  double t3832;
  double t4004;
  double t4040;
  double t4103;
  double t2345;
  double t2405;
  double t2452;
  double t2454;
  double t2658;
  double t2741;
  double t2755;
  double t2833;
  double t2853;
  double t2871;
  double t2889;
  double t2892;
  double t2928;
  double t3006;
  double t3022;
  double t3040;
  double t3389;
  double t3470;
  double t3565;
  double t3597;
  double t3723;
  double t3733;
  double t5175;
  double t4950;
  double t4966;
  double t4967;
  double t4978;
  double t5036;
  double t5102;
  double t5439;
  double t5465;
  double t5489;
  double t5324;
  double t5393;
  double t5403;
  double t5197;
  double t5222;
  double t5232;
  double t5247;
  double t5285;
  double t5296;
  double t5887;
  double t5895;
  double t5916;
  double t5850;
  double t5865;
  double t5866;
  double t6067;
  double t6072;
  double t6249;
  double t6285;
  double t6291;
  double t6298;
  double t6301;
  double t6309;
  double t6313;
  double t6323;
  double t6388;
  double t6393;
  double t6395;
  double t6401;
  double t6402;
  double t6408;
  double t6415;
  double t6426;
  double t6428;
  double t1048;
  double t6571;
  double t6577;
  double t6580;
  double t6549;
  double t6552;
  double t6553;
  double t6034;
  double t6044;
  double t6047;
  double t6073;
  double t6079;
  double t6100;
  double t6109;
  double t6116;
  double t6118;
  double t6662;
  double t6663;
  double t6600;
  double t6745;
  double t6746;
  double t6750;
  double t6790;
  double t6791;
  double t6792;
  double t6646;
  double t5275;
  double t5312;
  double t6835;
  double t6836;
  double t6842;
  double t6827;
  double t6829;
  double t6832;
  double t5568;
  double t5658;
  double t5701;
  double t5716;
  double t5789;
  double t5836;
  double t5880;
  double t5919;
  double t5922;
  double t5924;
  double t5926;
  double t5932;
  double t5941;
  double t5943;
  double t5945;
  double t5946;
  double t5948;
  double t5962;
  double t5965;
  double t5979;
  double t5980;
  double t5982;
  double t5983;
  double t6935;
  double t6936;
  double t6938;
  double t6971;
  double t6973;
  double t6974;
  double t6486;
  double t6488;
  double t7033;
  double t7034;
  double t7037;
  double t7029;
  double t7030;
  double t7031;
  double t6500;
  double t6516;
  double t6528;
  double t6541;
  double t6542;
  double t6547;
  double t6563;
  double t6584;
  double t6585;
  double t6588;
  double t6590;
  double t6601;
  double t6606;
  double t6607;
  double t6617;
  double t6631;
  double t6643;
  double t6648;
  double t6652;
  double t6653;
  double t6655;
  double t7155;
  double t7156;
  double t7165;
  double t7180;
  double t7182;
  double t7183;
  double t7240;
  double t7242;
  double t7243;
  double t7250;
  double t7253;
  double t7254;
  double t7297;
  double t7299;
  double t7301;
  double t7292;
  double t7293;
  double t7295;
  double t6786;
  double t6789;
  double t6794;
  double t6796;
  double t6798;
  double t6800;
  double t6801;
  double t6802;
  double t6803;
  double t6806;
  double t6807;
  double t6808;
  double t6809;
  double t6810;
  double t6814;
  double t6817;
  double t4976;
  double t5114;
  double t5558;
  double t5561;
  double t5627;
  double t5631;
  double t5427;
  double t5505;
  double t5551;
  double t5604;
  double t7405;
  double t7407;
  double t6334;
  double t3604;
  double t3771;
  double t3848;
  double t4173;
  double t4254;
  double t4270;
  double t4272;
  double t4284;
  double t4326;
  double t4331;
  double t4361;
  double t4404;
  double t4497;
  double t4516;
  double t4528;
  double t4683;
  double t4789;
  double t6968;
  double t6970;
  double t6975;
  double t6979;
  double t6986;
  double t6987;
  double t6991;
  double t6998;
  double t6999;
  double t7003;
  double t7006;
  double t7007;
  double t7012;
  double t7013;
  double t7014;
  double t7016;
  double t7481;
  double t7486;
  double t7488;
  double t7491;
  double t7497;
  double t7500;
  double t7503;
  double t7508;
  double t6477;
  double t6479;
  double t6499;
  double t6513;
  double t6514;
  double t6493;
  double t6494;
  double t6498;
  double t6511;
  double t6293;
  double t6294;
  double t6308;
  double t6324;
  double t6328;
  double t6332;
  double t6333;
  double t6337;
  double t6347;
  double t6350;
  double t6365;
  double t6366;
  double t6370;
  double t6371;
  double t6382;
  double t6383;
  double t7557;
  double t7558;
  double t7589;
  double t7590;
  double t7591;
  double t7598;
  double t7599;
  double t7600;
  double t7284;
  double t7285;
  double t7308;
  double t7315;
  double t7320;
  double t7323;
  double t7296;
  double t7302;
  double t7307;
  double t7319;
  double t7142;
  double t7147;
  double t7168;
  double t7186;
  double t7187;
  double t7190;
  double t7193;
  double t7194;
  double t7197;
  double t7201;
  double t7205;
  double t7209;
  double t7210;
  double t7219;
  double t7220;
  double t7224;
  double t7226;
  double t6741;
  double t6742;
  double t6751;
  double t6754;
  double t6756;
  double t6757;
  double t6758;
  double t6761;
  double t6762;
  double t6767;
  double t6769;
  double t6771;
  double t6777;
  double t6778;
  double t6779;
  double t6784;
  double t7403;
  double t7404;
  double t7415;
  double t7417;
  double t7418;
  double t7419;
  double t7425;
  double t7426;
  double t7427;
  double t7429;
  double t7430;
  double t7432;
  double t7435;
  double t7436;
  double t7437;
  double t7438;
  double t7723;
  double t7724;
  double t6445;
  double t542;
  double t604;
  double t1329;
  double t1643;
  double t1651;
  double t1982;
  double t2004;
  double t2122;
  double t2229;
  double t2407;
  double t2593;
  double t2605;
  double t2641;
  double t2884;
  double t3088;
  double t3149;
  double t6719;
  double t6931;
  double t6933;
  double t6939;
  double t6940;
  double t6941;
  double t6948;
  double t6949;
  double t6950;
  double t6952;
  double t6953;
  double t6955;
  double t6957;
  double t6958;
  double t6959;
  double t6963;
  double t6965;
  double t7479;
  double t7489;
  double t7501;
  double t7511;
  double t7512;
  double t7514;
  double t7517;
  double t7519;
  double t7520;
  double t7522;
  double t7523;
  double t7524;
  double t7525;
  double t7527;
  double t7528;
  double t7531;
  double t7532;
  double t7816;
  double t7818;
  double t7826;
  double t7827;
  double t6385;
  double t6399;
  double t6413;
  double t6432;
  double t6434;
  double t6437;
  double t6443;
  double t6446;
  double t6450;
  double t6452;
  double t6458;
  double t6459;
  double t6465;
  double t6466;
  double t6468;
  double t6470;
  double t7861;
  double t7587;
  double t7588;
  double t7595;
  double t7610;
  double t7618;
  double t7619;
  double t7620;
  double t7623;
  double t7624;
  double t7626;
  double t7627;
  double t7628;
  double t7631;
  double t7635;
  double t7637;
  double t7640;
  double t7643;
  double t7646;
  double t7649;
  double t7651;
  double t7658;
  double t7660;
  double t7661;
  double t7237;
  double t7238;
  double t7246;
  double t7255;
  double t7256;
  double t7257;
  double t7258;
  double t7259;
  double t7261;
  double t7263;
  double t7268;
  double t7269;
  double t7270;
  double t7271;
  double t7272;
  double t7279;
  double t7280;
  double t6878;
  double t6879;
  double t6880;
  double t6882;
  double t6888;
  double t6889;
  double t6893;
  double t6894;
  double t6900;
  double t6906;
  double t6907;
  double t6913;
  double t6915;
  double t6919;
  double t6921;
  double t6922;
  double t6924;
  double t6925;
  double t6927;
  double t7441;
  double t7445;
  double t7446;
  double t7447;
  double t7449;
  double t7450;
  double t7451;
  double t7454;
  double t7456;
  double t7460;
  double t7461;
  double t7463;
  double t7464;
  double t7471;
  double t7474;
  double t7780;
  double t7782;
  double t7784;
  double t7785;
  double t7789;
  double t7791;
  double t7793;
  double t7794;
  double t7798;
  double t7800;
  double t7803;
  double t7804;
  double t7805;
  double t7806;
  double t6062;
  double t6108;
  double t6119;
  double t6125;
  double t6133;
  double t6136;
  double t6139;
  double t6145;
  double t6146;
  double t6149;
  double t6161;
  double t6162;
  double t6165;
  double t6166;
  double t6168;
  double t6169;
  double t6178;
  double t6180;
  double t6201;
  double t7080;
  double t7083;
  double t7088;
  double t7099;
  double t7100;
  double t7102;
  double t7103;
  double t7104;
  double t7107;
  double t7108;
  double t7109;
  double t7115;
  double t7117;
  double t7119;
  double t7121;
  double t7123;
  double t7124;
  double t7128;
  double t7134;
  double t6660;
  double t6666;
  double t6676;
  double t6681;
  double t6685;
  double t6686;
  double t6687;
  double t6690;
  double t6692;
  double t6698;
  double t6704;
  double t6708;
  double t6711;
  double t6715;
  double t6727;
  double t6732;
  double t6737;
  double t6738;
  double t7559;
  double t7560;
  double t7561;
  double t7562;
  double t7564;
  double t7565;
  double t7567;
  double t7568;
  double t7570;
  double t7572;
  double t7573;
  double t7576;
  double t7579;
  double t7581;
  double t7582;
  double t7862;
  double t7864;
  double t7868;
  double t7869;
  double t7873;
  double t7878;
  double t7880;
  double t7881;
  double t7882;
  double t7884;
  double t7885;
  double t7886;
  double t7888;
  double t7889;
  double t7891;
  double t7949;
  double t7953;
  double t7954;
  double t7956;
  double t7960;
  double t7961;
  double t7964;
  double t7965;
  double t7967;
  double t8013;
  double t8014;
  double t7907;
  double t7909;
  double t7911;
  double t7912;
  double t7913;
  double t7918;
  double t7919;
  double t7924;
  double t7930;
  double t7931;
  double t7932;
  double t7934;
  double t7938;
  double t7939;
  double t7940;
  double t7671;
  double t7673;
  double t7674;
  double t7675;
  double t7681;
  double t7684;
  double t7686;
  double t7688;
  double t7690;
  double t7692;
  double t7693;
  double t7694;
  double t7699;
  double t7702;
  double t7707;
  double t7339;
  double t7340;
  double t7341;
  double t7342;
  double t7343;
  double t7346;
  double t7347;
  double t7348;
  double t7352;
  double t7357;
  double t7359;
  double t7360;
  double t7369;
  double t7371;
  double t7377;
  double t7378;
  double t7379;
  double t7380;
  double t7381;
  t479 = Cos(var1[3]);
  t259 = Cos(var1[5]);
  t316 = Sin(var1[3]);
  t370 = Sin(var1[4]);
  t482 = Sin(var1[5]);
  t256 = Cos(var1[6]);
  t564 = -1.*t479*t259;
  t565 = -1.*t316*t370*t482;
  t566 = t564 + t565;
  t412 = -1.*t259*t316*t370;
  t494 = t479*t482;
  t523 = t412 + t494;
  t580 = Sin(var1[6]);
  t668 = Cos(var1[7]);
  t679 = -1.*t668;
  t737 = 1. + t679;
  t778 = t256*t566;
  t795 = -1.*t523*t580;
  t1019 = t778 + t795;
  t1156 = -1.*t256*t523;
  t1214 = -1.*t566*t580;
  t1308 = t1156 + t1214;
  t2232 = Sin(var1[7]);
  t3691 = t479*t259;
  t3702 = t316*t370*t482;
  t3716 = t3691 + t3702;
  t1102 = -0.8656776547239999*t737;
  t1116 = 1. + t1102;
  t1811 = -0.134322983001*t737;
  t1866 = 1. + t1811;
  t3798 = t256*t3716;
  t3821 = t523*t580;
  t3832 = t3798 + t3821;
  t4004 = t256*t523;
  t4040 = -1.*t3716*t580;
  t4103 = t4004 + t4040;
  t2345 = -0.930418*t2232;
  t2405 = 0. + t2345;
  t2452 = -0.366501*t2232;
  t2454 = 0. + t2452;
  t2658 = -3.2909349868922137e-7*var1[7];
  t2741 = 0.03103092645718495*t737;
  t2755 = 0.366501*t2232;
  t2833 = 0. + t2755;
  t2853 = -0.045000372235*t2833;
  t2871 = t2658 + t2741 + t2853;
  t2889 = 1.296332362046933e-7*var1[7];
  t2892 = 0.07877668146182712*t737;
  t2928 = 0.930418*t2232;
  t3006 = 0. + t2928;
  t3022 = -0.045000372235*t3006;
  t3040 = t2889 + t2892 + t3022;
  t3389 = -1.*t256;
  t3470 = 1. + t3389;
  t3565 = 0.091*t3470;
  t3597 = 0. + t3565;
  t3723 = 0.091*t580;
  t3733 = 0. + t3723;
  t5175 = Cos(var1[4]);
  t4950 = t259*t316;
  t4966 = -1.*t479*t370*t482;
  t4967 = t4950 + t4966;
  t4978 = -1.*t479*t259*t370;
  t5036 = -1.*t316*t482;
  t5102 = t4978 + t5036;
  t5439 = t256*t5102;
  t5465 = t4967*t580;
  t5489 = t5439 + t5465;
  t5324 = t256*t4967;
  t5393 = -1.*t5102*t580;
  t5403 = t5324 + t5393;
  t5197 = -0.04500040093286238*t737;
  t5222 = 0.07877663122399998*t2405;
  t5232 = 0.031030906668*t2454;
  t5247 = 0. + t5197 + t5222 + t5232;
  t5285 = -1.000000637725*t737;
  t5296 = 1. + t5285;
  t5887 = -1.*t5175*t259*t256*t316;
  t5895 = -1.*t5175*t316*t482*t580;
  t5916 = t5887 + t5895;
  t5850 = -1.*t5175*t256*t316*t482;
  t5865 = t5175*t259*t316*t580;
  t5866 = t5850 + t5865;
  t6067 = t566*t580;
  t6072 = t4004 + t6067;
  t6249 = t479*t259*t370;
  t6285 = t316*t482;
  t6291 = t6249 + t6285;
  t6298 = t6291*t580;
  t6301 = t5324 + t6298;
  t6309 = t256*t6291;
  t6313 = -1.*t4967*t580;
  t6323 = t6309 + t6313;
  t6388 = -1.*t259*t316;
  t6393 = t479*t370*t482;
  t6395 = t6388 + t6393;
  t6401 = t256*t6395;
  t6402 = -1.*t6291*t580;
  t6408 = t6401 + t6402;
  t6415 = -1.*t256*t6291;
  t6426 = -1.*t6395*t580;
  t6428 = t6415 + t6426;
  t1048 = -0.340999127418*t737*t1019;
  t6571 = t479*t5175*t259*t256;
  t6577 = t479*t5175*t482*t580;
  t6580 = t6571 + t6577;
  t6549 = t479*t5175*t256*t482;
  t6552 = -1.*t479*t5175*t259*t580;
  t6553 = t6549 + t6552;
  t6034 = -0.0846680539949003*t668;
  t6044 = -0.04500040093286238*t2232;
  t6047 = t6034 + t6044;
  t6073 = -0.016492681424499736*t668;
  t6079 = 0.03103092645718495*t2232;
  t6100 = -3.2909349868922137e-7 + t6073 + t6079;
  t6109 = -0.04186915633414423*t668;
  t6116 = 0.07877668146182712*t2232;
  t6118 = 1.296332362046933e-7 + t6109 + t6116;
  t6662 = t6395*t580;
  t6663 = t6309 + t6662;
  t6600 = -0.340999127418*t737*t6553;
  t6745 = -1.*t479*t5175*t259*t256;
  t6746 = -1.*t479*t5175*t482*t580;
  t6750 = t6745 + t6746;
  t6790 = -1.*t479*t5175*t256*t482;
  t6791 = t479*t5175*t259*t580;
  t6792 = t6790 + t6791;
  t6646 = -0.340999127418*t737*t6580;
  t5275 = -1.*t479*t5175*t5247;
  t5312 = -1.*t479*t5175*t5296;
  t6835 = -1.*t479*t259*t256*t370;
  t6836 = -1.*t479*t370*t482*t580;
  t6842 = t6835 + t6836;
  t6827 = -1.*t479*t256*t370*t482;
  t6829 = t479*t259*t370*t580;
  t6832 = t6827 + t6829;
  t5568 = -1.*t479*t5175*t2833;
  t5658 = -1.*t479*t5175*t3006;
  t5701 = -1.*t5175*t3597*t316*t482;
  t5716 = -1.*t5175*t259*t316*t3733;
  t5789 = t316*t370*t5247;
  t5836 = t5296*t316*t370;
  t5880 = t5866*t2405;
  t5919 = t5916*t2454;
  t5922 = t5836 + t5880 + t5919;
  t5924 = -0.04501*t5922;
  t5926 = t5916*t2871;
  t5932 = -0.340999127418*t737*t5866;
  t5941 = t1866*t5916;
  t5943 = t316*t370*t2833;
  t5945 = t5932 + t5941 + t5943;
  t5946 = -0.086806*t5945;
  t5948 = t5866*t3040;
  t5962 = t1116*t5866;
  t5965 = -0.340999127418*t737*t5916;
  t5979 = t316*t370*t3006;
  t5980 = t5962 + t5965 + t5979;
  t5982 = 0.123098*t5980;
  t5983 = t5701 + t5716 + t5789 + t5924 + t5926 + t5946 + t5948 + t5982;
  t6935 = t5175*t256*t316*t482;
  t6936 = -1.*t5175*t259*t316*t580;
  t6938 = t6935 + t6936;
  t6971 = t5175*t259*t256*t316;
  t6973 = t5175*t316*t482*t580;
  t6974 = t6971 + t6973;
  t6486 = -1.*t5175*t316*t5247;
  t6488 = -1.*t5175*t5296*t316;
  t7033 = -1.*t259*t256*t316*t370;
  t7034 = -1.*t316*t370*t482*t580;
  t7037 = t7033 + t7034;
  t7029 = -1.*t256*t316*t370*t482;
  t7030 = t259*t316*t370*t580;
  t7031 = t7029 + t7030;
  t6500 = -1.*t5175*t316*t2833;
  t6516 = -1.*t5175*t316*t3006;
  t6528 = t479*t5175*t3597*t482;
  t6541 = t479*t5175*t259*t3733;
  t6542 = -1.*t479*t370*t5247;
  t6547 = -1.*t479*t5296*t370;
  t6563 = t6553*t2405;
  t6584 = t6580*t2454;
  t6585 = t6547 + t6563 + t6584;
  t6588 = -0.04501*t6585;
  t6590 = t6580*t2871;
  t6601 = t1866*t6580;
  t6606 = -1.*t479*t370*t2833;
  t6607 = t6600 + t6601 + t6606;
  t6617 = -0.086806*t6607;
  t6631 = t6553*t3040;
  t6643 = t1116*t6553;
  t6648 = -1.*t479*t370*t3006;
  t6652 = t6643 + t6646 + t6648;
  t6653 = 0.123098*t6652;
  t6655 = t6528 + t6541 + t6542 + t6588 + t6590 + t6617 + t6631 + t6653;
  t7155 = t256*t370*t482;
  t7156 = -1.*t259*t370*t580;
  t7165 = t7155 + t7156;
  t7180 = -1.*t259*t256*t370;
  t7182 = -1.*t370*t482*t580;
  t7183 = t7180 + t7182;
  t7240 = -1.*t256*t370*t482;
  t7242 = t259*t370*t580;
  t7243 = t7240 + t7242;
  t7250 = t259*t256*t370;
  t7253 = t370*t482*t580;
  t7254 = t7250 + t7253;
  t7297 = -1.*t5175*t259*t256;
  t7299 = -1.*t5175*t482*t580;
  t7301 = t7297 + t7299;
  t7292 = -1.*t5175*t256*t482;
  t7293 = t5175*t259*t580;
  t7295 = t7292 + t7293;
  t6786 = t479*t5175*t259*t3597;
  t6789 = -1.*t479*t5175*t482*t3733;
  t6794 = -0.340999127418*t737*t6792;
  t6796 = t1116*t6580;
  t6798 = t6794 + t6796;
  t6800 = 0.123098*t6798;
  t6801 = t1866*t6792;
  t6802 = t6801 + t6646;
  t6803 = -0.086806*t6802;
  t6806 = t6580*t2405;
  t6807 = t6792*t2454;
  t6808 = t6806 + t6807;
  t6809 = -0.04501*t6808;
  t6810 = t6792*t2871;
  t6814 = t6580*t3040;
  t6817 = t6786 + t6789 + t6800 + t6803 + t6809 + t6810 + t6814;
  t4976 = t3597*t4967;
  t5114 = t5102*t3733;
  t5558 = -0.340999127418*t737*t5403;
  t5561 = t1866*t5489;
  t5627 = t1116*t5403;
  t5631 = -0.340999127418*t737*t5489;
  t5427 = t5403*t2405;
  t5505 = t5489*t2454;
  t5551 = t5489*t2871;
  t5604 = t5403*t3040;
  t7405 = -1.*t256*t4967;
  t7407 = t7405 + t6402;
  t6334 = -0.340999127418*t737*t6323;
  t3604 = t3597*t523;
  t3771 = t3716*t3733;
  t3848 = -0.340999127418*t737*t3832;
  t4173 = t1116*t4103;
  t4254 = t3848 + t4173;
  t4270 = 0.123098*t4254;
  t4272 = t1866*t3832;
  t4284 = -0.340999127418*t737*t4103;
  t4326 = t4272 + t4284;
  t4331 = -0.086806*t4326;
  t4361 = t4103*t2405;
  t4404 = t3832*t2454;
  t4497 = t4361 + t4404;
  t4516 = -0.04501*t4497;
  t4528 = t3832*t2871;
  t4683 = t4103*t3040;
  t4789 = t3604 + t3771 + t4270 + t4331 + t4516 + t4528 + t4683;
  t6968 = t5175*t259*t3597*t316;
  t6970 = -1.*t5175*t316*t482*t3733;
  t6975 = t1116*t6974;
  t6979 = t5932 + t6975;
  t6986 = 0.123098*t6979;
  t6987 = t1866*t5866;
  t6991 = -0.340999127418*t737*t6974;
  t6998 = t6987 + t6991;
  t6999 = -0.086806*t6998;
  t7003 = t6974*t2405;
  t7006 = t5866*t2454;
  t7007 = t7003 + t7006;
  t7012 = -0.04501*t7007;
  t7013 = t5866*t2871;
  t7014 = t6974*t3040;
  t7016 = t6968 + t6970 + t6986 + t6999 + t7012 + t7013 + t7014;
  t7481 = t259*t316*t370;
  t7486 = -1.*t479*t482;
  t7488 = t7481 + t7486;
  t7491 = -1.*t256*t566;
  t7497 = -1.*t7488*t580;
  t7500 = t7491 + t7497;
  t7503 = t256*t7488;
  t7508 = t7503 + t1214;
  t6477 = t3597*t566;
  t6479 = t523*t3733;
  t6499 = t1866*t6072;
  t6513 = t1116*t1019;
  t6514 = -0.340999127418*t737*t6072;
  t6493 = t1019*t2405;
  t6494 = t6072*t2454;
  t6498 = t6072*t2871;
  t6511 = t1019*t3040;
  t6293 = t3597*t6291;
  t6294 = t4967*t3733;
  t6308 = -0.340999127418*t737*t6301;
  t6324 = t1116*t6323;
  t6328 = t6308 + t6324;
  t6332 = 0.123098*t6328;
  t6333 = t1866*t6301;
  t6337 = t6333 + t6334;
  t6347 = -0.086806*t6337;
  t6350 = t6323*t2405;
  t6365 = t6301*t2454;
  t6366 = t6350 + t6365;
  t6370 = -0.04501*t6366;
  t6371 = t6301*t2871;
  t6382 = t6323*t3040;
  t6383 = t6293 + t6294 + t6332 + t6347 + t6370 + t6371 + t6382;
  t7557 = t7488*t580;
  t7558 = t778 + t7557;
  t7589 = t5175*t256*t482;
  t7590 = -1.*t5175*t259*t580;
  t7591 = t7589 + t7590;
  t7598 = t5175*t259*t256;
  t7599 = t5175*t482*t580;
  t7600 = t7598 + t7599;
  t7284 = -1.*t5175*t3597*t482;
  t7285 = -1.*t5175*t259*t3733;
  t7308 = -0.340999127418*t737*t7295;
  t7315 = t1866*t7301;
  t7320 = t1116*t7295;
  t7323 = -0.340999127418*t737*t7301;
  t7296 = t7295*t2405;
  t7302 = t7301*t2454;
  t7307 = t7301*t2871;
  t7319 = t7295*t3040;
  t7142 = -1.*t259*t3597*t370;
  t7147 = t370*t482*t3733;
  t7168 = -0.340999127418*t737*t7165;
  t7186 = t1116*t7183;
  t7187 = t7168 + t7186;
  t7190 = 0.123098*t7187;
  t7193 = t1866*t7165;
  t7194 = -0.340999127418*t737*t7183;
  t7197 = t7193 + t7194;
  t7201 = -0.086806*t7197;
  t7205 = t7183*t2405;
  t7209 = t7165*t2454;
  t7210 = t7205 + t7209;
  t7219 = -0.04501*t7210;
  t7220 = t7165*t2871;
  t7224 = t7183*t3040;
  t7226 = t7142 + t7147 + t7190 + t7201 + t7219 + t7220 + t7224;
  t6741 = 0.091*t479*t5175*t259*t256;
  t6742 = 0.091*t479*t5175*t482*t580;
  t6751 = t1116*t6750;
  t6754 = t6600 + t6751;
  t6756 = 0.123098*t6754;
  t6757 = t1866*t6553;
  t6758 = -0.340999127418*t737*t6750;
  t6761 = t6757 + t6758;
  t6762 = -0.086806*t6761;
  t6767 = t6750*t2405;
  t6769 = t6553*t2454;
  t6771 = t6767 + t6769;
  t6777 = -0.04501*t6771;
  t6778 = t6553*t2871;
  t6779 = t6750*t3040;
  t6784 = t6741 + t6742 + t6756 + t6762 + t6777 + t6778 + t6779;
  t7403 = 0.091*t256*t4967;
  t7404 = 0.091*t6291*t580;
  t7415 = -0.340999127418*t737*t7407;
  t7417 = t1866*t6323;
  t7418 = t7415 + t7417;
  t7419 = -0.086806*t7418;
  t7425 = t1116*t7407;
  t7426 = t7425 + t6334;
  t7427 = 0.123098*t7426;
  t7429 = t7407*t2405;
  t7430 = t6323*t2454;
  t7432 = t7429 + t7430;
  t7435 = -0.04501*t7432;
  t7436 = t6323*t2871;
  t7437 = t7407*t3040;
  t7438 = t7403 + t7404 + t7419 + t7427 + t7435 + t7436 + t7437;
  t7723 = -1.*t256*t6395;
  t7724 = t7723 + t6298;
  t6445 = -0.340999127418*t737*t6428;
  t542 = 0.091*t256*t523;
  t604 = 0.091*t566*t580;
  t1329 = t1116*t1308;
  t1643 = t1048 + t1329;
  t1651 = 0.123098*t1643;
  t1982 = t1866*t1019;
  t2004 = -0.340999127418*t737*t1308;
  t2122 = t1982 + t2004;
  t2229 = -0.086806*t2122;
  t2407 = t1308*t2405;
  t2593 = t1019*t2454;
  t2605 = t2407 + t2593;
  t2641 = -0.04501*t2605;
  t2884 = t1019*t2871;
  t3088 = t1308*t3040;
  t3149 = t542 + t604 + t1651 + t2229 + t2641 + t2884 + t3088;
  t6719 = -0.340999127418*t6408*t2232;
  t6931 = 0.091*t5175*t259*t256*t316;
  t6933 = 0.091*t5175*t316*t482*t580;
  t6939 = -0.340999127418*t737*t6938;
  t6940 = t1116*t5916;
  t6941 = t6939 + t6940;
  t6948 = 0.123098*t6941;
  t6949 = t1866*t6938;
  t6950 = t6949 + t5965;
  t6952 = -0.086806*t6950;
  t6953 = t5916*t2405;
  t6955 = t6938*t2454;
  t6957 = t6953 + t6955;
  t6958 = -0.04501*t6957;
  t6959 = t6938*t2871;
  t6963 = t5916*t3040;
  t6965 = t6931 + t6933 + t6948 + t6952 + t6958 + t6959 + t6963;
  t7479 = 0.091*t256*t566;
  t7489 = 0.091*t7488*t580;
  t7501 = -0.340999127418*t737*t7500;
  t7511 = t1866*t7508;
  t7512 = t7501 + t7511;
  t7514 = -0.086806*t7512;
  t7517 = t1116*t7500;
  t7519 = -0.340999127418*t737*t7508;
  t7520 = t7517 + t7519;
  t7522 = 0.123098*t7520;
  t7523 = t7500*t2405;
  t7524 = t7508*t2454;
  t7525 = t7523 + t7524;
  t7527 = -0.04501*t7525;
  t7528 = t7508*t2871;
  t7531 = t7500*t3040;
  t7532 = t7479 + t7489 + t7514 + t7522 + t7527 + t7528 + t7531;
  t7816 = -1.*t256*t3716;
  t7818 = t7816 + t7557;
  t7826 = -1.*t256*t7488;
  t7827 = t7826 + t4040;
  t6385 = 0.091*t256*t6291;
  t6399 = 0.091*t6395*t580;
  t6413 = -0.340999127418*t737*t6408;
  t6432 = t1116*t6428;
  t6434 = t6413 + t6432;
  t6437 = 0.123098*t6434;
  t6443 = t1866*t6408;
  t6446 = t6443 + t6445;
  t6450 = -0.086806*t6446;
  t6452 = t6428*t2405;
  t6458 = t6408*t2454;
  t6459 = t6452 + t6458;
  t6465 = -0.04501*t6459;
  t6466 = t6408*t2871;
  t6468 = t6428*t3040;
  t6470 = t6385 + t6399 + t6437 + t6450 + t6465 + t6466 + t6468;
  t7861 = t3798 + t7497;
  t7587 = -0.091*t5175*t256*t482;
  t7588 = 0.091*t5175*t259*t580;
  t7595 = -0.340999127418*t737*t7591;
  t7610 = t1866*t7600;
  t7618 = t7595 + t7610;
  t7619 = -0.086806*t7618;
  t7620 = t1116*t7591;
  t7623 = -0.340999127418*t737*t7600;
  t7624 = t7620 + t7623;
  t7626 = 0.123098*t7624;
  t7627 = t7591*t2405;
  t7628 = t7600*t2454;
  t7631 = t7627 + t7628;
  t7635 = -0.04501*t7631;
  t7637 = t7600*t2871;
  t7640 = t7591*t3040;
  t7643 = t7587 + t7588 + t7619 + t7626 + t7635 + t7637 + t7640;
  t7646 = t7308 + t7315;
  t7649 = -0.086806*t7646;
  t7651 = t7320 + t7323;
  t7658 = 0.123098*t7651;
  t7660 = t7296 + t7302;
  t7661 = -0.04501*t7660;
  t7237 = -0.091*t259*t256*t370;
  t7238 = -0.091*t370*t482*t580;
  t7246 = -0.340999127418*t737*t7243;
  t7255 = t1116*t7254;
  t7256 = t7246 + t7255;
  t7257 = 0.123098*t7256;
  t7258 = t1866*t7243;
  t7259 = -0.340999127418*t737*t7254;
  t7261 = t7258 + t7259;
  t7263 = -0.086806*t7261;
  t7268 = t7254*t2405;
  t7269 = t7243*t2454;
  t7270 = t7268 + t7269;
  t7271 = -0.04501*t7270;
  t7272 = t7243*t2871;
  t7279 = t7254*t3040;
  t7280 = t7237 + t7238 + t7257 + t7263 + t7271 + t7272 + t7279;
  t6878 = -1.*t479*t370*t6047;
  t6879 = t6580*t6100;
  t6880 = t6553*t6118;
  t6882 = -0.930418*t668*t6553;
  t6888 = -0.366501*t668*t6580;
  t6889 = 1.000000637725*t479*t370*t2232;
  t6893 = t6882 + t6888 + t6889;
  t6894 = -0.04501*t6893;
  t6900 = -0.930418*t479*t668*t370;
  t6906 = -0.8656776547239999*t6553*t2232;
  t6907 = -0.340999127418*t6580*t2232;
  t6913 = t6900 + t6906 + t6907;
  t6915 = 0.123098*t6913;
  t6919 = -0.366501*t479*t668*t370;
  t6921 = -0.340999127418*t6553*t2232;
  t6922 = -0.134322983001*t6580*t2232;
  t6924 = t6919 + t6921 + t6922;
  t6925 = -0.086806*t6924;
  t6927 = t6878 + t6879 + t6880 + t6894 + t6915 + t6925;
  t7441 = -0.366501*t668*t6301;
  t7445 = -0.930418*t668*t6323;
  t7446 = t7441 + t7445;
  t7447 = -0.04501*t7446;
  t7449 = t6301*t6100;
  t7450 = t6323*t6118;
  t7451 = -0.340999127418*t6301*t2232;
  t7454 = -0.8656776547239999*t6323*t2232;
  t7456 = t7451 + t7454;
  t7460 = 0.123098*t7456;
  t7461 = -0.134322983001*t6301*t2232;
  t7463 = -0.340999127418*t6323*t2232;
  t7464 = t7461 + t7463;
  t7471 = -0.086806*t7464;
  t7474 = t7447 + t7449 + t7450 + t7460 + t7471;
  t7780 = -0.366501*t668*t6408;
  t7782 = -0.930418*t668*t6428;
  t7784 = t7780 + t7782;
  t7785 = -0.04501*t7784;
  t7789 = t6408*t6100;
  t7791 = t6428*t6118;
  t7793 = -0.8656776547239999*t6428*t2232;
  t7794 = t6719 + t7793;
  t7798 = 0.123098*t7794;
  t7800 = -0.134322983001*t6408*t2232;
  t7803 = -0.340999127418*t6428*t2232;
  t7804 = t7800 + t7803;
  t7805 = -0.086806*t7804;
  t7806 = t7785 + t7789 + t7791 + t7798 + t7805;
  t6062 = -1.*t5175*t316*t6047;
  t6108 = t6072*t6100;
  t6119 = t1019*t6118;
  t6125 = -0.930418*t668*t1019;
  t6133 = -0.366501*t668*t6072;
  t6136 = 1.000000637725*t5175*t316*t2232;
  t6139 = t6125 + t6133 + t6136;
  t6145 = -0.04501*t6139;
  t6146 = -0.930418*t5175*t668*t316;
  t6149 = -0.8656776547239999*t1019*t2232;
  t6161 = -0.340999127418*t6072*t2232;
  t6162 = t6146 + t6149 + t6161;
  t6165 = 0.123098*t6162;
  t6166 = -0.366501*t5175*t668*t316;
  t6168 = -0.340999127418*t1019*t2232;
  t6169 = -0.134322983001*t6072*t2232;
  t6178 = t6166 + t6168 + t6169;
  t6180 = -0.086806*t6178;
  t6201 = t6062 + t6108 + t6119 + t6145 + t6165 + t6180;
  t7080 = -1.*t316*t370*t6047;
  t7083 = t6974*t6100;
  t7088 = t6938*t6118;
  t7099 = -0.930418*t668*t6938;
  t7100 = -0.366501*t668*t6974;
  t7102 = 1.000000637725*t316*t370*t2232;
  t7103 = t7099 + t7100 + t7102;
  t7104 = -0.04501*t7103;
  t7107 = -0.930418*t668*t316*t370;
  t7108 = -0.8656776547239999*t6938*t2232;
  t7109 = -0.340999127418*t6974*t2232;
  t7115 = t7107 + t7108 + t7109;
  t7117 = 0.123098*t7115;
  t7119 = -0.366501*t668*t316*t370;
  t7121 = -0.340999127418*t6938*t2232;
  t7123 = -0.134322983001*t6974*t2232;
  t7124 = t7119 + t7121 + t7123;
  t7128 = -0.086806*t7124;
  t7134 = t7080 + t7083 + t7088 + t7104 + t7117 + t7128;
  t6660 = t479*t5175*t6047;
  t6666 = t6663*t6100;
  t6676 = t6408*t6118;
  t6681 = -0.930418*t668*t6408;
  t6685 = -0.366501*t668*t6663;
  t6686 = -1.000000637725*t479*t5175*t2232;
  t6687 = t6681 + t6685 + t6686;
  t6690 = -0.04501*t6687;
  t6692 = 0.930418*t479*t5175*t668;
  t6698 = -0.8656776547239999*t6408*t2232;
  t6704 = -0.340999127418*t6663*t2232;
  t6708 = t6692 + t6698 + t6704;
  t6711 = 0.123098*t6708;
  t6715 = 0.366501*t479*t5175*t668;
  t6727 = -0.134322983001*t6663*t2232;
  t6732 = t6715 + t6719 + t6727;
  t6737 = -0.086806*t6732;
  t6738 = t6660 + t6666 + t6676 + t6690 + t6711 + t6737;
  t7559 = -0.366501*t668*t7558;
  t7560 = -0.930418*t668*t7508;
  t7561 = t7559 + t7560;
  t7562 = -0.04501*t7561;
  t7564 = t7558*t6100;
  t7565 = t7508*t6118;
  t7567 = -0.340999127418*t7558*t2232;
  t7568 = -0.8656776547239999*t7508*t2232;
  t7570 = t7567 + t7568;
  t7572 = 0.123098*t7570;
  t7573 = -0.134322983001*t7558*t2232;
  t7576 = -0.340999127418*t7508*t2232;
  t7579 = t7573 + t7576;
  t7581 = -0.086806*t7579;
  t7582 = t7562 + t7564 + t7565 + t7572 + t7581;
  t7862 = -0.366501*t668*t7861;
  t7864 = -0.930418*t668*t7827;
  t7868 = t7862 + t7864;
  t7869 = -0.04501*t7868;
  t7873 = t7861*t6100;
  t7878 = t7827*t6118;
  t7880 = -0.340999127418*t7861*t2232;
  t7881 = -0.8656776547239999*t7827*t2232;
  t7882 = t7880 + t7881;
  t7884 = 0.123098*t7882;
  t7885 = -0.134322983001*t7861*t2232;
  t7886 = -0.340999127418*t7827*t2232;
  t7888 = t7885 + t7886;
  t7889 = -0.086806*t7888;
  t7891 = t7869 + t7873 + t7878 + t7884 + t7889;
  t7949 = 0.03103092645718495*t668;
  t7953 = 0.016492681424499736*t2232;
  t7954 = t7949 + t7953;
  t7956 = 0.07877668146182712*t668;
  t7960 = 0.04186915633414423*t2232;
  t7961 = t7956 + t7960;
  t7964 = -0.04500040093286238*t668;
  t7965 = 0.0846680539949003*t2232;
  t7967 = t7964 + t7965;
  t8013 = t3716*t580;
  t8014 = t7503 + t8013;
  t7907 = -0.366501*t668*t7591;
  t7909 = -0.930418*t668*t7301;
  t7911 = t7907 + t7909;
  t7912 = -0.04501*t7911;
  t7913 = t7591*t6100;
  t7918 = t7301*t6118;
  t7919 = -0.340999127418*t7591*t2232;
  t7924 = -0.8656776547239999*t7301*t2232;
  t7930 = t7919 + t7924;
  t7931 = 0.123098*t7930;
  t7932 = -0.134322983001*t7591*t2232;
  t7934 = -0.340999127418*t7301*t2232;
  t7938 = t7932 + t7934;
  t7939 = -0.086806*t7938;
  t7940 = t7912 + t7913 + t7918 + t7931 + t7939;
  t7671 = -0.366501*t668*t7295;
  t7673 = -0.930418*t668*t7600;
  t7674 = t7671 + t7673;
  t7675 = -0.04501*t7674;
  t7681 = t7295*t6100;
  t7684 = t7600*t6118;
  t7686 = -0.340999127418*t7295*t2232;
  t7688 = -0.8656776547239999*t7600*t2232;
  t7690 = t7686 + t7688;
  t7692 = 0.123098*t7690;
  t7693 = -0.134322983001*t7295*t2232;
  t7694 = -0.340999127418*t7600*t2232;
  t7699 = t7693 + t7694;
  t7702 = -0.086806*t7699;
  t7707 = t7675 + t7681 + t7684 + t7692 + t7702;
  t7339 = -1.*t5175*t6047;
  t7340 = t7183*t6100;
  t7341 = t7243*t6118;
  t7342 = -0.930418*t668*t7243;
  t7343 = -0.366501*t668*t7183;
  t7346 = 1.000000637725*t5175*t2232;
  t7347 = t7342 + t7343 + t7346;
  t7348 = -0.04501*t7347;
  t7352 = -0.930418*t5175*t668;
  t7357 = -0.8656776547239999*t7243*t2232;
  t7359 = -0.340999127418*t7183*t2232;
  t7360 = t7352 + t7357 + t7359;
  t7369 = 0.123098*t7360;
  t7371 = -0.366501*t5175*t668;
  t7377 = -0.340999127418*t7243*t2232;
  t7378 = -0.134322983001*t7183*t2232;
  t7379 = t7371 + t7377 + t7378;
  t7380 = -0.086806*t7379;
  t7381 = t7339 + t7340 + t7341 + t7348 + t7369 + t7380;
  p_output1[0]=0;
  p_output1[1]=0;
  p_output1[2]=0;
  p_output1[3]=0;
  p_output1[4]=0;
  p_output1[5]=0;
  p_output1[6]=0;
  p_output1[7]=0;
  p_output1[8]=0;
  p_output1[9]=(t4976 + t5114 + t5275 - 0.04501*(t5312 + t5427 + t5505) + t5551 - 0.086806*(t5558 + t5561 + t5568) + t5604 + 0.123098*(t5627 + t5631 + t5658))*var2[3] + t5983*var2[4] + t4789*var2[5] + t3149*var2[6] + t6201*var2[7];
  p_output1[10]=(t6477 + t6479 + t6486 - 0.04501*(t6488 + t6493 + t6494) + t6498 - 0.086806*(t1048 + t6499 + t6500) + t6511 + 0.123098*(t6513 + t6514 + t6516))*var2[3] + t6655*var2[4] + t6383*var2[5] + t6470*var2[6] + t6738*var2[7];
  p_output1[11]=0;
  p_output1[12]=t5983*var2[3] + (-1.*t259*t370*t3733*t479 - 1.*t3597*t370*t479*t482 + t5275 + t3040*t6832 + t2871*t6842 - 0.04501*(t5312 + t2405*t6832 + t2454*t6842) - 0.086806*(t5568 + t1866*t6842 - 0.340999127418*t6832*t737) + 0.123098*(t5658 + t1116*t6832 - 0.340999127418*t6842*t737))*var2[4] + t6817*var2[5] + t6784*var2[6] + t6927*var2[7];
  p_output1[13]=t6655*var2[3] + (-1.*t259*t316*t370*t3733 - 1.*t316*t3597*t370*t482 + t6486 + t3040*t7031 + t2871*t7037 - 0.04501*(t6488 + t2405*t7031 + t2454*t7037) - 0.086806*(t6500 + t1866*t7037 - 0.340999127418*t7031*t737) + 0.123098*(t6516 + t1116*t7031 - 0.340999127418*t7037*t737))*var2[4] + t7016*var2[5] + t6965*var2[6] + t7134*var2[7];
  p_output1[14]=(t370*t5247 + t7284 + t7285 - 0.04501*(t370*t5296 + t7296 + t7302) + t7307 - 0.086806*(t2833*t370 + t7308 + t7315) + t7319 + 0.123098*(t3006*t370 + t7320 + t7323))*var2[4] + t7226*var2[5] + t7280*var2[6] + t7381*var2[7];
  p_output1[15]=t4789*var2[3] + t6817*var2[4] + (t4976 + t5114 - 0.04501*(t5427 + t5505) + t5551 - 0.086806*(t5558 + t5561) + t5604 + 0.123098*(t5627 + t5631))*var2[5] + t7438*var2[6] + t7474*var2[7];
  p_output1[16]=t6383*var2[3] + t7016*var2[4] + (t6477 + t6479 - 0.04501*(t6493 + t6494) + t6498 - 0.086806*(t1048 + t6499) + t6511 + 0.123098*(t6513 + t6514))*var2[5] + t7532*var2[6] + t7582*var2[7];
  p_output1[17]=t7226*var2[4] + (t7284 + t7285 + t7307 + t7319 + t7649 + t7658 + t7661)*var2[5] + t7643*var2[6] + t7707*var2[7];
  p_output1[18]=t3149*var2[3] + t6784*var2[4] + t7438*var2[5] + (-0.091*t580*t6291 + 0.091*t256*t6395 + t2871*t6428 + t3040*t7724 + 0.123098*(t6445 + t1116*t7724) - 0.04501*(t2454*t6428 + t2405*t7724) - 0.086806*(t1866*t6428 - 0.340999127418*t737*t7724))*var2[6] + t7806*var2[7];
  p_output1[19]=t6470*var2[3] + t6965*var2[4] + t7532*var2[5] + (0.091*t256*t3716 - 0.091*t580*t7488 + t3040*t7818 + t2871*t7827 - 0.086806*(-0.340999127418*t737*t7818 + t1866*t7827) - 0.04501*(t2405*t7818 + t2454*t7827) + 0.123098*(t1116*t7818 - 0.340999127418*t737*t7827))*var2[6] + t7891*var2[7];
  p_output1[20]=t7280*var2[4] + t7643*var2[5] + (0.091*t256*t482*t5175 - 0.091*t259*t5175*t580 + t7307 + t7319 + t7649 + t7658 + t7661)*var2[6] + t7940*var2[7];
  p_output1[21]=t6201*var2[3] + t6927*var2[4] + t7474*var2[5] + t7806*var2[6] + (-0.04501*(0.930418*t2232*t6408 + 0.366501*t2232*t6663 - 1.000000637725*t479*t5175*t668) + 0.123098*(-0.930418*t2232*t479*t5175 - 0.8656776547239999*t6408*t668 - 0.340999127418*t6663*t668) - 0.086806*(-0.366501*t2232*t479*t5175 - 0.340999127418*t6408*t668 - 0.134322983001*t6663*t668) + t6663*t7954 + t6408*t7961 + t479*t5175*t7967)*var2[7];
  p_output1[22]=t6738*var2[3] + t7134*var2[4] + t7582*var2[5] + t7891*var2[6] + (t7861*t7961 + t316*t5175*t7967 + t7954*t8014 - 0.04501*(-1.000000637725*t316*t5175*t668 + 0.930418*t2232*t7861 + 0.366501*t2232*t8014) + 0.123098*(-0.930418*t2232*t316*t5175 - 0.8656776547239999*t668*t7861 - 0.340999127418*t668*t8014) - 0.086806*(-0.366501*t2232*t316*t5175 - 0.340999127418*t668*t7861 - 0.134322983001*t668*t8014))*var2[7];
  p_output1[23]=t7381*var2[4] + t7707*var2[5] + t7940*var2[6] + (-0.04501*(1.000000637725*t370*t668 + 0.930418*t2232*t7591 + 0.366501*t2232*t7600) + 0.123098*(0.930418*t2232*t370 - 0.8656776547239999*t668*t7591 - 0.340999127418*t668*t7600) - 0.086806*(0.366501*t2232*t370 - 0.340999127418*t668*t7591 - 0.134322983001*t668*t7600) + t7600*t7954 + t7591*t7961 - 1.*t370*t7967)*var2[7];
  p_output1[24]=0;
  p_output1[25]=0;
  p_output1[26]=0;
  p_output1[27]=0;
  p_output1[28]=0;
  p_output1[29]=0;
  p_output1[30]=0;
  p_output1[31]=0;
  p_output1[32]=0;
  p_output1[33]=0;
  p_output1[34]=0;
  p_output1[35]=0;
  p_output1[36]=0;
  p_output1[37]=0;
  p_output1[38]=0;
  p_output1[39]=0;
  p_output1[40]=0;
  p_output1[41]=0;
  p_output1[42]=0;
  p_output1[43]=0;
  p_output1[44]=0;
  p_output1[45]=0;
  p_output1[46]=0;
  p_output1[47]=0;
  p_output1[48]=0;
  p_output1[49]=0;
  p_output1[50]=0;
  p_output1[51]=0;
  p_output1[52]=0;
  p_output1[53]=0;
  p_output1[54]=0;
  p_output1[55]=0;
  p_output1[56]=0;
  p_output1[57]=0;
  p_output1[58]=0;
  p_output1[59]=0;
  p_output1[60]=0;
  p_output1[61]=0;
  p_output1[62]=0;
  p_output1[63]=0;
  p_output1[64]=0;
  p_output1[65]=0;
  p_output1[66]=0;
  p_output1[67]=0;
  p_output1[68]=0;
  p_output1[69]=0;
  p_output1[70]=0;
  p_output1[71]=0;
  p_output1[72]=0;
  p_output1[73]=0;
  p_output1[74]=0;
  p_output1[75]=0;
  p_output1[76]=0;
  p_output1[77]=0;
  p_output1[78]=0;
  p_output1[79]=0;
  p_output1[80]=0;
  p_output1[81]=0;
  p_output1[82]=0;
  p_output1[83]=0;
}



void dJp_left_hip_yaw_src(double *p_output1, const double *var1,const double *var2)
{
  // Call Subroutines
  output1(p_output1, var1, var2);

}

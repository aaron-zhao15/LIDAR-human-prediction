/*
 * Automatically Generated from Mathematica.
 * Tue 17 Aug 2021 12:17:43 GMT-04:00
 */

#ifdef MATLAB_MEX_FILE
#include <stdexcept>
#include <cmath>
#include<math.h>
/**
 * Copied from Wolfram Mathematica C Definitions file mdefs.hpp
 * Changed marcos to inline functions (Eric Cousineau)
 */
inline double Power(double x, double y) { return pow(x, y); }
inline double Sqrt(double x) { return sqrt(x); }

inline double Abs(double x) { return fabs(x); }

inline double Exp(double x) { return exp(x); }
inline double Log(double x) { return log(x); }

inline double Sin(double x) { return sin(x); }
inline double Cos(double x) { return cos(x); }
inline double Tan(double x) { return tan(x); }

inline double ArcSin(double x) { return asin(x); }
inline double ArcCos(double x) { return acos(x); }
inline double ArcTan(double x) { return atan(x); }

/* update ArcTan function to use atan2 instead. */
inline double ArcTan(double x, double y) { return atan2(y,x); }

inline double Sinh(double x) { return sinh(x); }
inline double Cosh(double x) { return cosh(x); }
inline double Tanh(double x) { return tanh(x); }

const double E	= 2.71828182845904523536029;
const double Pi = 3.14159265358979323846264;
const double Degree = 0.01745329251994329576924;

inline double Sec(double x) { return 1/cos(x); }
inline double Csc(double x) { return 1/sin(x); }

#endif
#include "T_leftFoot.hh"

/*
 * Sub functions
 */
static void output1(double *p_output1,const double *var1)
{
  double t3454;
  double t3837;
  double t3815;
  double t3825;
  double t3826;
  double t3848;
  double t3839;
  double t3841;
  double t3849;
  double t3802;
  double t3804;
  double t3847;
  double t3850;
  double t3869;
  double t3875;
  double t3883;
  double t3898;
  double t3948;
  double t3949;
  double t3952;
  double t3807;
  double t3871;
  double t3872;
  double t3873;
  double t3901;
  double t3903;
  double t3909;
  double t3922;
  double t3928;
  double t3938;
  double t3939;
  double t3944;
  double t3958;
  double t3971;
  double t3972;
  double t3974;
  double t3978;
  double t3980;
  double t3989;
  double t3990;
  double t3991;
  double t3794;
  double t3800;
  double t3906;
  double t3946;
  double t3953;
  double t3954;
  double t3981;
  double t3982;
  double t3995;
  double t3999;
  double t4005;
  double t4007;
  double t4008;
  double t4010;
  double t4022;
  double t4023;
  double t4026;
  double t4027;
  double t4042;
  double t4043;
  double t4066;
  double t4067;
  double t4068;
  double t3988;
  double t4014;
  double t4016;
  double t4017;
  double t4046;
  double t4048;
  double t4054;
  double t4055;
  double t4056;
  double t4057;
  double t4060;
  double t4061;
  double t4075;
  double t4077;
  double t4078;
  double t4079;
  double t4082;
  double t4083;
  double t4090;
  double t4091;
  double t4092;
  double t3764;
  double t3772;
  double t4049;
  double t4065;
  double t4073;
  double t4074;
  double t4084;
  double t4086;
  double t4093;
  double t4094;
  double t4095;
  double t4098;
  double t4100;
  double t4101;
  double t4106;
  double t4107;
  double t4109;
  double t4112;
  double t4114;
  double t4117;
  double t3573;
  double t3577;
  double t3595;
  double t3601;
  double t4135;
  double t4136;
  double t4137;
  double t4087;
  double t4102;
  double t4103;
  double t4105;
  double t4119;
  double t4120;
  double t4124;
  double t4125;
  double t4126;
  double t4131;
  double t4132;
  double t4133;
  double t4141;
  double t4142;
  double t4143;
  double t4147;
  double t4148;
  double t4151;
  double t4121;
  double t4134;
  double t4138;
  double t4139;
  double t4152;
  double t4154;
  double t4162;
  double t4163;
  double t4167;
  double t4168;
  double t4169;
  double t4170;
  double t3599;
  double t4174;
  double t4175;
  double t4176;
  double t4177;
  double t4179;
  double t4181;
  double t4195;
  double t4158;
  double t3604;
  double t3642;
  double t4223;
  double t4226;
  double t4225;
  double t4227;
  double t4228;
  double t4230;
  double t4231;
  double t4232;
  double t4229;
  double t4233;
  double t4236;
  double t4238;
  double t4239;
  double t4242;
  double t4224;
  double t4237;
  double t4243;
  double t4244;
  double t4246;
  double t4247;
  double t4248;
  double t4249;
  double t4252;
  double t4253;
  double t4254;
  double t4256;
  double t4245;
  double t4251;
  double t4257;
  double t4262;
  double t4264;
  double t4265;
  double t4266;
  double t4267;
  double t4270;
  double t4271;
  double t4272;
  double t4273;
  double t4263;
  double t4268;
  double t4274;
  double t4275;
  double t4277;
  double t4278;
  double t4280;
  double t4281;
  double t4284;
  double t4285;
  double t4286;
  double t4289;
  double t4276;
  double t4282;
  double t4290;
  double t4291;
  double t4294;
  double t4295;
  double t4296;
  double t4297;
  double t4299;
  double t4300;
  double t4301;
  double t4302;
  double t4159;
  double t4160;
  double t4293;
  double t4298;
  double t4303;
  double t4304;
  double t4307;
  double t4309;
  double t4310;
  double t4311;
  double t4316;
  double t4317;
  double t4321;
  double t4324;
  double t4172;
  double t4173;
  double t4188;
  double t4189;
  double t4305;
  double t4314;
  double t4325;
  double t4327;
  double t4196;
  double t4197;
  double t4329;
  double t4330;
  double t4332;
  double t4333;
  double t4200;
  double t4201;
  double t4335;
  double t4336;
  double t4337;
  double t4338;
  double t4206;
  double t4207;
  double t4210;
  double t4211;
  double t4213;
  double t4215;
  double t4359;
  double t4360;
  double t4361;
  double t4363;
  double t4365;
  double t4366;
  double t4362;
  double t4367;
  double t4368;
  double t4370;
  double t4371;
  double t4372;
  double t4358;
  double t4369;
  double t4373;
  double t4374;
  double t4377;
  double t4378;
  double t4379;
  double t4380;
  double t4382;
  double t4383;
  double t4384;
  double t4385;
  double t4376;
  double t4381;
  double t4386;
  double t4387;
  double t4390;
  double t4391;
  double t4394;
  double t4395;
  double t4398;
  double t4399;
  double t4400;
  double t4401;
  double t4389;
  double t4396;
  double t4402;
  double t4403;
  double t4405;
  double t4406;
  double t4407;
  double t4408;
  double t4410;
  double t4412;
  double t4413;
  double t4414;
  double t4404;
  double t4409;
  double t4415;
  double t4417;
  double t4419;
  double t4422;
  double t4424;
  double t4426;
  double t4428;
  double t4429;
  double t4430;
  double t4431;
  double t4418;
  double t4427;
  double t4432;
  double t4433;
  double t4435;
  double t4436;
  double t4437;
  double t4441;
  double t4443;
  double t4444;
  double t4448;
  double t4449;
  double t4434;
  double t4442;
  double t4450;
  double t4451;
  double t4453;
  double t4454;
  double t4455;
  double t4456;
  double t4458;
  double t4459;
  double t4460;
  double t4461;
  double t4156;
  double t4171;
  double t4182;
  double t4183;
  double t4186;
  double t4193;
  double t4199;
  double t4202;
  double t4204;
  double t4205;
  double t4208;
  double t4212;
  double t4216;
  double t4217;
  double t4219;
  double t4221;
  double t4328;
  double t4334;
  double t4339;
  double t4340;
  double t4341;
  double t4346;
  double t4347;
  double t4348;
  double t4349;
  double t4350;
  double t4351;
  double t4352;
  double t4353;
  double t4354;
  double t4355;
  double t4356;
  double t4452;
  double t4457;
  double t4462;
  double t4463;
  double t4465;
  double t4466;
  double t4467;
  double t4468;
  double t4469;
  double t4470;
  double t4471;
  double t4472;
  double t4473;
  double t4474;
  double t4475;
  double t4476;
  double t4487;
  double t4488;
  double t4489;
  double t4491;
  double t4493;
  double t4494;
  double t4496;
  double t4497;
  double t4498;
  double t4509;
  double t4510;
  double t4511;
  double t4512;
  double t4514;
  double t4515;
  double t4516;
  double t4517;
  double t4519;
  double t4521;
  double t4522;
  double t4523;
  double t4638;
  double t4647;
  double t4635;
  double t4539;
  double t4540;
  double t4541;
  double t4543;
  double t4544;
  double t4546;
  double t4547;
  double t4548;
  double t4549;
  double t4551;
  double t4553;
  double t4554;
  double t4555;
  double t4557;
  double t4558;
  double t4560;
  double t4561;
  double t4563;
  double t4565;
  double t4566;
  double t4569;
  double t4571;
  double t4572;
  double t4573;
  double t4575;
  double t4576;
  double t4577;
  double t4579;
  double t4580;
  double t4581;
  double t4582;
  double t4584;
  double t4585;
  double t4586;
  double t4587;
  double t4589;
  double t4590;
  double t4591;
  double t4592;
  double t4595;
  double t4596;
  double t4597;
  double t4598;
  double t4600;
  double t4601;
  double t4602;
  double t4604;
  double t4605;
  double t4606;
  double t4608;
  double t4609;
  double t4610;
  double t4611;
  double t4613;
  double t4614;
  double t4615;
  double t4616;
  double t4618;
  double t4619;
  double t4620;
  double t4621;
  double t4624;
  double t4625;
  double t4626;
  double t4627;
  double t4629;
  double t4630;
  double t4631;
  double t4633;
  double t4634;
  double t4636;
  double t4637;
  double t4639;
  double t4640;
  double t4641;
  double t4643;
  double t4644;
  double t4645;
  double t4646;
  double t4648;
  double t4649;
  double t4650;
  double t4652;
  double t4653;
  double t4654;
  double t4655;
  double t4656;
  double t4657;
  double t4658;
  double t4538;
  double t4542;
  double t4545;
  double t4550;
  double t4556;
  double t4562;
  double t4570;
  double t4574;
  double t4578;
  double t4583;
  double t4588;
  double t4593;
  double t4599;
  double t4603;
  double t4607;
  double t4612;
  double t4617;
  double t4623;
  double t4628;
  double t4632;
  double t4642;
  double t4651;
  double t4659;
  double t4660;
  double t4661;
  double t4662;
  double t4663;
  double t4665;
  double t4667;
  double t4668;
  double t4669;
  double t4670;
  double t4671;
  double t4672;
  double t4674;
  double t4675;
  double t4676;
  double t4679;
  double t4680;
  double t4681;
  double t4682;
  double t4683;
  double t4684;
  double t4685;
  double t4686;
  double t4687;
  double t4688;
  double t4689;
  double t4690;
  double t4691;
  double t4692;
  double t4693;
  double t4694;
  double t4695;
  double t4697;
  double t4698;
  double t4699;
  double t4700;
  double t4701;
  double t4702;
  double t4703;
  double t4706;
  double t4707;
  double t4708;
  double t4709;
  double t4710;
  double t4711;
  double t4712;
  double t4713;
  double t4714;
  double t4715;
  double t4716;
  double t4717;
  double t4718;
  double t4719;
  double t4720;
  double t4721;
  double t4722;
  double t4723;
  double t4724;
  double t4725;
  t3454 = Sin(var1[4]);
  t3837 = Cos(var1[4]);
  t3815 = Cos(var1[15]);
  t3825 = -1.*t3815;
  t3826 = 1. + t3825;
  t3848 = Cos(var1[14]);
  t3839 = Cos(var1[5]);
  t3841 = Sin(var1[14]);
  t3849 = Sin(var1[5]);
  t3802 = Sin(var1[16]);
  t3804 = Sin(var1[15]);
  t3847 = -1.*t3837*t3839*t3841;
  t3850 = t3848*t3837*t3849;
  t3869 = t3847 + t3850;
  t3875 = t3848*t3837*t3839;
  t3883 = t3837*t3841*t3849;
  t3898 = t3875 + t3883;
  t3948 = Cos(var1[16]);
  t3949 = -1.*t3948;
  t3952 = 1. + t3949;
  t3807 = 0.366501*t3804*t3454;
  t3871 = 0.340999127418*t3826*t3869;
  t3872 = -0.134322983001*t3826;
  t3873 = 1. + t3872;
  t3901 = t3873*t3898;
  t3903 = t3807 + t3871 + t3901;
  t3909 = -0.930418*t3804*t3454;
  t3922 = -0.8656776547239999*t3826;
  t3928 = 1. + t3922;
  t3938 = t3928*t3869;
  t3939 = 0.340999127418*t3826*t3898;
  t3944 = t3909 + t3938 + t3939;
  t3958 = -1.000000637725*t3826;
  t3971 = 1. + t3958;
  t3972 = -1.*t3971*t3454;
  t3974 = -0.930418*t3804*t3869;
  t3978 = 0.366501*t3804*t3898;
  t3980 = t3972 + t3974 + t3978;
  t3989 = Cos(var1[17]);
  t3990 = -1.*t3989;
  t3991 = 1. + t3990;
  t3794 = Sin(var1[18]);
  t3800 = Sin(var1[17]);
  t3906 = 0.930418*t3802*t3903;
  t3946 = 0.366501*t3802*t3944;
  t3953 = -1.000000637725*t3952;
  t3954 = 1. + t3953;
  t3981 = t3954*t3980;
  t3982 = t3906 + t3946 + t3981;
  t3995 = -0.8656776547239999*t3952;
  t3999 = 1. + t3995;
  t4005 = t3999*t3903;
  t4007 = -0.340999127418*t3952*t3944;
  t4008 = -0.930418*t3802*t3980;
  t4010 = t4005 + t4007 + t4008;
  t4022 = -0.340999127418*t3952*t3903;
  t4023 = -0.134322983001*t3952;
  t4026 = 1. + t4023;
  t4027 = t4026*t3944;
  t4042 = -0.366501*t3802*t3980;
  t4043 = t4022 + t4027 + t4042;
  t4066 = Cos(var1[18]);
  t4067 = -1.*t4066;
  t4068 = 1. + t4067;
  t3988 = 0.366501*t3800*t3982;
  t4014 = -0.340999127418*t3991*t4010;
  t4016 = -0.134322983001*t3991;
  t4017 = 1. + t4016;
  t4046 = t4017*t4043;
  t4048 = t3988 + t4014 + t4046;
  t4054 = 0.930418*t3800*t3982;
  t4055 = -0.8656776547239999*t3991;
  t4056 = 1. + t4055;
  t4057 = t4056*t4010;
  t4060 = -0.340999127418*t3991*t4043;
  t4061 = t4054 + t4057 + t4060;
  t4075 = -1.000000637725*t3991;
  t4077 = 1. + t4075;
  t4078 = t4077*t3982;
  t4079 = -0.930418*t3800*t4010;
  t4082 = -0.366501*t3800*t4043;
  t4083 = t4078 + t4079 + t4082;
  t4090 = Cos(var1[19]);
  t4091 = -1.*t4090;
  t4092 = 1. + t4091;
  t3764 = Sin(var1[20]);
  t3772 = Sin(var1[19]);
  t4049 = -0.366501*t3794*t4048;
  t4065 = -0.930418*t3794*t4061;
  t4073 = -1.000000637725*t4068;
  t4074 = 1. + t4073;
  t4084 = t4074*t4083;
  t4086 = t4049 + t4065 + t4084;
  t4093 = -0.134322983001*t4068;
  t4094 = 1. + t4093;
  t4095 = t4094*t4048;
  t4098 = -0.340999127418*t4068*t4061;
  t4100 = 0.366501*t3794*t4083;
  t4101 = t4095 + t4098 + t4100;
  t4106 = -0.340999127418*t4068*t4048;
  t4107 = -0.8656776547239999*t4068;
  t4109 = 1. + t4107;
  t4112 = t4109*t4061;
  t4114 = 0.930418*t3794*t4083;
  t4117 = t4106 + t4112 + t4114;
  t3573 = Cos(var1[21]);
  t3577 = -1.*t3573;
  t3595 = 1. + t3577;
  t3601 = Sin(var1[21]);
  t4135 = Cos(var1[20]);
  t4136 = -1.*t4135;
  t4137 = 1. + t4136;
  t4087 = 0.930418*t3772*t4086;
  t4102 = -0.340999127418*t4092*t4101;
  t4103 = -0.8656776547239999*t4092;
  t4105 = 1. + t4103;
  t4119 = t4105*t4117;
  t4120 = t4087 + t4102 + t4119;
  t4124 = 0.366501*t3772*t4086;
  t4125 = -0.134322983001*t4092;
  t4126 = 1. + t4125;
  t4131 = t4126*t4101;
  t4132 = -0.340999127418*t4092*t4117;
  t4133 = t4124 + t4131 + t4132;
  t4141 = -1.000000637725*t4092;
  t4142 = 1. + t4141;
  t4143 = t4142*t4086;
  t4147 = -0.366501*t3772*t4101;
  t4148 = -0.930418*t3772*t4117;
  t4151 = t4143 + t4147 + t4148;
  t4121 = -0.930418*t3764*t4120;
  t4134 = -0.366501*t3764*t4133;
  t4138 = -1.000000637725*t4137;
  t4139 = 1. + t4138;
  t4152 = t4139*t4151;
  t4154 = t4121 + t4134 + t4152;
  t4162 = -0.340999127418*t4137*t4120;
  t4163 = -0.134322983001*t4137;
  t4167 = 1. + t4163;
  t4168 = t4167*t4133;
  t4169 = 0.366501*t3764*t4151;
  t4170 = t4162 + t4168 + t4169;
  t3599 = 0.444895486988*t3595;
  t4174 = -0.8656776547239999*t4137;
  t4175 = 1. + t4174;
  t4176 = t4175*t4120;
  t4177 = -0.340999127418*t4137*t4133;
  t4179 = 0.930418*t3764*t4151;
  t4181 = t4176 + t4177 + t4179;
  t4195 = 0.175248972904*t3595;
  t4158 = 0.120666640478*t3595;
  t3604 = 0.218018*t3601;
  t3642 = t3599 + t3604;
  t4223 = Cos(var1[3]);
  t4226 = Sin(var1[3]);
  t4225 = t4223*t3839*t3454;
  t4227 = t4226*t3849;
  t4228 = t4225 + t4227;
  t4230 = -1.*t3839*t4226;
  t4231 = t4223*t3454*t3849;
  t4232 = t4230 + t4231;
  t4229 = -1.*t3841*t4228;
  t4233 = t3848*t4232;
  t4236 = t4229 + t4233;
  t4238 = t3848*t4228;
  t4239 = t3841*t4232;
  t4242 = t4238 + t4239;
  t4224 = -0.366501*t4223*t3837*t3804;
  t4237 = 0.340999127418*t3826*t4236;
  t4243 = t3873*t4242;
  t4244 = t4224 + t4237 + t4243;
  t4246 = 0.930418*t4223*t3837*t3804;
  t4247 = t3928*t4236;
  t4248 = 0.340999127418*t3826*t4242;
  t4249 = t4246 + t4247 + t4248;
  t4252 = t3971*t4223*t3837;
  t4253 = -0.930418*t3804*t4236;
  t4254 = 0.366501*t3804*t4242;
  t4256 = t4252 + t4253 + t4254;
  t4245 = 0.930418*t3802*t4244;
  t4251 = 0.366501*t3802*t4249;
  t4257 = t3954*t4256;
  t4262 = t4245 + t4251 + t4257;
  t4264 = t3999*t4244;
  t4265 = -0.340999127418*t3952*t4249;
  t4266 = -0.930418*t3802*t4256;
  t4267 = t4264 + t4265 + t4266;
  t4270 = -0.340999127418*t3952*t4244;
  t4271 = t4026*t4249;
  t4272 = -0.366501*t3802*t4256;
  t4273 = t4270 + t4271 + t4272;
  t4263 = 0.366501*t3800*t4262;
  t4268 = -0.340999127418*t3991*t4267;
  t4274 = t4017*t4273;
  t4275 = t4263 + t4268 + t4274;
  t4277 = 0.930418*t3800*t4262;
  t4278 = t4056*t4267;
  t4280 = -0.340999127418*t3991*t4273;
  t4281 = t4277 + t4278 + t4280;
  t4284 = t4077*t4262;
  t4285 = -0.930418*t3800*t4267;
  t4286 = -0.366501*t3800*t4273;
  t4289 = t4284 + t4285 + t4286;
  t4276 = -0.366501*t3794*t4275;
  t4282 = -0.930418*t3794*t4281;
  t4290 = t4074*t4289;
  t4291 = t4276 + t4282 + t4290;
  t4294 = t4094*t4275;
  t4295 = -0.340999127418*t4068*t4281;
  t4296 = 0.366501*t3794*t4289;
  t4297 = t4294 + t4295 + t4296;
  t4299 = -0.340999127418*t4068*t4275;
  t4300 = t4109*t4281;
  t4301 = 0.930418*t3794*t4289;
  t4302 = t4299 + t4300 + t4301;
  t4159 = -0.803828*t3601;
  t4160 = t4158 + t4159;
  t4293 = 0.930418*t3772*t4291;
  t4298 = -0.340999127418*t4092*t4297;
  t4303 = t4105*t4302;
  t4304 = t4293 + t4298 + t4303;
  t4307 = 0.366501*t3772*t4291;
  t4309 = t4126*t4297;
  t4310 = -0.340999127418*t4092*t4302;
  t4311 = t4307 + t4309 + t4310;
  t4316 = t4142*t4291;
  t4317 = -0.366501*t3772*t4297;
  t4321 = -0.930418*t3772*t4302;
  t4324 = t4316 + t4317 + t4321;
  t4172 = -0.693671301908*t3595;
  t4173 = 1. + t4172;
  t4188 = -0.353861996165*t3595;
  t4189 = 1. + t4188;
  t4305 = -0.930418*t3764*t4304;
  t4314 = -0.366501*t3764*t4311;
  t4325 = t4139*t4324;
  t4327 = t4305 + t4314 + t4325;
  t4196 = 0.553471*t3601;
  t4197 = t4195 + t4196;
  t4329 = -0.340999127418*t4137*t4304;
  t4330 = t4167*t4311;
  t4332 = 0.366501*t3764*t4324;
  t4333 = t4329 + t4330 + t4332;
  t4200 = -0.218018*t3601;
  t4201 = t3599 + t4200;
  t4335 = t4175*t4304;
  t4336 = -0.340999127418*t4137*t4311;
  t4337 = 0.930418*t3764*t4324;
  t4338 = t4335 + t4336 + t4337;
  t4206 = -0.553471*t3601;
  t4207 = t4195 + t4206;
  t4210 = -0.952469601425*t3595;
  t4211 = 1. + t4210;
  t4213 = 0.803828*t3601;
  t4215 = t4158 + t4213;
  t4359 = t3839*t4226*t3454;
  t4360 = -1.*t4223*t3849;
  t4361 = t4359 + t4360;
  t4363 = t4223*t3839;
  t4365 = t4226*t3454*t3849;
  t4366 = t4363 + t4365;
  t4362 = -1.*t3841*t4361;
  t4367 = t3848*t4366;
  t4368 = t4362 + t4367;
  t4370 = t3848*t4361;
  t4371 = t3841*t4366;
  t4372 = t4370 + t4371;
  t4358 = -0.366501*t3837*t3804*t4226;
  t4369 = 0.340999127418*t3826*t4368;
  t4373 = t3873*t4372;
  t4374 = t4358 + t4369 + t4373;
  t4377 = 0.930418*t3837*t3804*t4226;
  t4378 = t3928*t4368;
  t4379 = 0.340999127418*t3826*t4372;
  t4380 = t4377 + t4378 + t4379;
  t4382 = t3971*t3837*t4226;
  t4383 = -0.930418*t3804*t4368;
  t4384 = 0.366501*t3804*t4372;
  t4385 = t4382 + t4383 + t4384;
  t4376 = 0.930418*t3802*t4374;
  t4381 = 0.366501*t3802*t4380;
  t4386 = t3954*t4385;
  t4387 = t4376 + t4381 + t4386;
  t4390 = t3999*t4374;
  t4391 = -0.340999127418*t3952*t4380;
  t4394 = -0.930418*t3802*t4385;
  t4395 = t4390 + t4391 + t4394;
  t4398 = -0.340999127418*t3952*t4374;
  t4399 = t4026*t4380;
  t4400 = -0.366501*t3802*t4385;
  t4401 = t4398 + t4399 + t4400;
  t4389 = 0.366501*t3800*t4387;
  t4396 = -0.340999127418*t3991*t4395;
  t4402 = t4017*t4401;
  t4403 = t4389 + t4396 + t4402;
  t4405 = 0.930418*t3800*t4387;
  t4406 = t4056*t4395;
  t4407 = -0.340999127418*t3991*t4401;
  t4408 = t4405 + t4406 + t4407;
  t4410 = t4077*t4387;
  t4412 = -0.930418*t3800*t4395;
  t4413 = -0.366501*t3800*t4401;
  t4414 = t4410 + t4412 + t4413;
  t4404 = -0.366501*t3794*t4403;
  t4409 = -0.930418*t3794*t4408;
  t4415 = t4074*t4414;
  t4417 = t4404 + t4409 + t4415;
  t4419 = t4094*t4403;
  t4422 = -0.340999127418*t4068*t4408;
  t4424 = 0.366501*t3794*t4414;
  t4426 = t4419 + t4422 + t4424;
  t4428 = -0.340999127418*t4068*t4403;
  t4429 = t4109*t4408;
  t4430 = 0.930418*t3794*t4414;
  t4431 = t4428 + t4429 + t4430;
  t4418 = 0.930418*t3772*t4417;
  t4427 = -0.340999127418*t4092*t4426;
  t4432 = t4105*t4431;
  t4433 = t4418 + t4427 + t4432;
  t4435 = 0.366501*t3772*t4417;
  t4436 = t4126*t4426;
  t4437 = -0.340999127418*t4092*t4431;
  t4441 = t4435 + t4436 + t4437;
  t4443 = t4142*t4417;
  t4444 = -0.366501*t3772*t4426;
  t4448 = -0.930418*t3772*t4431;
  t4449 = t4443 + t4444 + t4448;
  t4434 = -0.930418*t3764*t4433;
  t4442 = -0.366501*t3764*t4441;
  t4450 = t4139*t4449;
  t4451 = t4434 + t4442 + t4450;
  t4453 = -0.340999127418*t4137*t4433;
  t4454 = t4167*t4441;
  t4455 = 0.366501*t3764*t4449;
  t4456 = t4453 + t4454 + t4455;
  t4458 = t4175*t4433;
  t4459 = -0.340999127418*t4137*t4441;
  t4460 = 0.930418*t3764*t4449;
  t4461 = t4458 + t4459 + t4460;
  t4156 = t3642*t4154;
  t4171 = t4160*t4170;
  t4182 = t4173*t4181;
  t4183 = t4156 + t4171 + t4182;
  t4186 = 0.105372*t4183;
  t4193 = t4189*t4154;
  t4199 = t4197*t4170;
  t4202 = t4201*t4181;
  t4204 = t4193 + t4199 + t4202;
  t4205 = 0.993566*t4204;
  t4208 = t4207*t4154;
  t4212 = t4211*t4170;
  t4216 = t4215*t4181;
  t4217 = t4208 + t4212 + t4216;
  t4219 = 0.041507*t4217;
  t4221 = t4186 + t4205 + t4219;
  t4328 = t3642*t4327;
  t4334 = t4160*t4333;
  t4339 = t4173*t4338;
  t4340 = t4328 + t4334 + t4339;
  t4341 = 0.105372*t4340;
  t4346 = t4189*t4327;
  t4347 = t4197*t4333;
  t4348 = t4201*t4338;
  t4349 = t4346 + t4347 + t4348;
  t4350 = 0.993566*t4349;
  t4351 = t4207*t4327;
  t4352 = t4211*t4333;
  t4353 = t4215*t4338;
  t4354 = t4351 + t4352 + t4353;
  t4355 = 0.041507*t4354;
  t4356 = t4341 + t4350 + t4355;
  t4452 = t3642*t4451;
  t4457 = t4160*t4456;
  t4462 = t4173*t4461;
  t4463 = t4452 + t4457 + t4462;
  t4465 = 0.105372*t4463;
  t4466 = t4189*t4451;
  t4467 = t4197*t4456;
  t4468 = t4201*t4461;
  t4469 = t4466 + t4467 + t4468;
  t4470 = 0.993566*t4469;
  t4471 = t4207*t4451;
  t4472 = t4211*t4456;
  t4473 = t4215*t4461;
  t4474 = t4471 + t4472 + t4473;
  t4475 = 0.041507*t4474;
  t4476 = t4465 + t4470 + t4475;
  t4487 = -0.366501*t4183;
  t4488 = 0.930418*t4217;
  t4489 = t4487 + t4488;
  t4491 = -0.366501*t4340;
  t4493 = 0.930418*t4354;
  t4494 = t4491 + t4493;
  t4496 = -0.366501*t4463;
  t4497 = 0.930418*t4474;
  t4498 = t4496 + t4497;
  t4509 = 0.924432*t4183;
  t4510 = -0.113252*t4204;
  t4511 = 0.364143*t4217;
  t4512 = t4509 + t4510 + t4511;
  t4514 = 0.924432*t4340;
  t4515 = -0.113252*t4349;
  t4516 = 0.364143*t4354;
  t4517 = t4514 + t4515 + t4516;
  t4519 = 0.924432*t4463;
  t4521 = -0.113252*t4469;
  t4522 = 0.364143*t4474;
  t4523 = t4519 + t4521 + t4522;
  t4638 = -0.175248972904*t3595;
  t4647 = -0.120666640478*t3595;
  t4635 = -0.444895486988*t3595;
  t4539 = -0.04500040093286238*t3826;
  t4540 = 0.0846680539949003*t3804;
  t4541 = t4539 + t4540;
  t4543 = -1.*t3848;
  t4544 = 1. + t4543;
  t4546 = 1.296332362046933e-7*var1[15];
  t4547 = -0.07877668146182712*t3826;
  t4548 = -0.04186915633414423*t3804;
  t4549 = t4546 + t4547 + t4548;
  t4551 = 3.2909349868922137e-7*var1[15];
  t4553 = 0.03103092645718495*t3826;
  t4554 = 0.016492681424499736*t3804;
  t4555 = t4551 + t4553 + t4554;
  t4557 = -1.296332362046933e-7*var1[16];
  t4558 = -0.14128592423750855*t3952;
  t4560 = 0.04186915633414423*t3802;
  t4561 = t4557 + t4558 + t4560;
  t4563 = 3.2909349868922137e-7*var1[16];
  t4565 = -0.055653945343889656*t3952;
  t4566 = 0.016492681424499736*t3802;
  t4569 = t4563 + t4565 + t4566;
  t4571 = -0.04500040093286238*t3952;
  t4572 = -0.15185209683981668*t3802;
  t4573 = t4571 + t4572;
  t4575 = 0.039853038461262744*t3991;
  t4576 = 0.23670515095269612*t3800;
  t4577 = t4575 + t4576;
  t4579 = 6.295460977284962e-8*var1[17];
  t4580 = -0.22023473313910558*t3991;
  t4581 = 0.03707996069223323*t3800;
  t4582 = t4579 + t4580 + t4581;
  t4584 = -1.5981976069815686e-7*var1[17];
  t4585 = -0.08675267452931407*t3991;
  t4586 = 0.014606169134372047*t3800;
  t4587 = t4584 + t4585 + t4586;
  t4589 = -4.0833068682577724e-7*var1[18];
  t4590 = -0.11476729583292707*t4068;
  t4591 = 0.0111594154470601*t3794;
  t4592 = t4589 + t4590 + t4591;
  t4595 = 1.6084556086870008e-7*var1[18];
  t4596 = -0.29135406957765553*t4068;
  t4597 = 0.02832985722118838*t3794;
  t4598 = t4595 + t4596 + t4597;
  t4600 = 0.03044854601678662*t4068;
  t4601 = 0.3131431996991197*t3794;
  t4602 = t4600 + t4601;
  t4604 = -0.26285954081199375*t4092;
  t4605 = 0.634735404786378*t3772;
  t4606 = t4604 + t4605;
  t4608 = 1.6169269214444473e-7*var1[19];
  t4609 = -0.2326311605896123*t4092;
  t4610 = -0.09633822312984319*t3772;
  t4611 = t4608 + t4609 + t4610;
  t4613 = -6.369237629068993e-8*var1[19];
  t4614 = -0.5905692458505322*t4092;
  t4615 = -0.24456909227538925*t3772;
  t4616 = t4613 + t4614 + t4615;
  t4618 = -7.041766963257243e-8*var1[20];
  t4619 = -0.8232948486053725*t4137;
  t4620 = 0.05763710717422546*t3764;
  t4621 = t4618 + t4619 + t4620;
  t4624 = 1.7876586242383724e-7*var1[20];
  t4625 = -0.3243041141817093*t4137;
  t4626 = 0.02270383571304597*t3764;
  t4627 = t4624 + t4625 + t4626;
  t4629 = 0.06194758047549556*t4137;
  t4630 = 0.8848655643005321*t3764;
  t4631 = t4629 + t4630;
  t4633 = 2.7989049814696287e-7*var1[21];
  t4634 = 0.15748067958019524*t3595;
  t4636 = t4635 + t4200;
  t4637 = -0.528674719304*t4636;
  t4639 = t4638 + t4196;
  t4640 = -0.29871295412*t4639;
  t4641 = t4633 + t4634 + t4637 + t4640;
  t4643 = 7.591321355439789e-8*var1[21];
  t4644 = -0.2845150083511607*t3595;
  t4645 = t4638 + t4206;
  t4646 = 0.445034169498*t4645;
  t4648 = t4647 + t4213;
  t4649 = -0.528674719304*t4648;
  t4650 = t4643 + t4644 + t4646 + t4649;
  t4652 = 1.9271694180831932e-7*var1[21];
  t4653 = -0.3667264808254521*t3595;
  t4654 = t4647 + t4159;
  t4655 = -0.29871295412*t4654;
  t4656 = t4635 + t3604;
  t4657 = 0.445034169498*t4656;
  t4658 = t4652 + t4653 + t4655 + t4657;
  t4538 = -0.091*t3837*t3839*t3841;
  t4542 = -1.*t4541*t3454;
  t4545 = -0.091*t4544*t3837*t3849;
  t4550 = t4549*t3869;
  t4556 = t4555*t3898;
  t4562 = t4561*t3903;
  t4570 = t4569*t3944;
  t4574 = t4573*t3980;
  t4578 = t4577*t3982;
  t4583 = t4582*t4010;
  t4588 = t4587*t4043;
  t4593 = t4592*t4048;
  t4599 = t4598*t4061;
  t4603 = t4602*t4083;
  t4607 = t4606*t4086;
  t4612 = t4611*t4101;
  t4617 = t4616*t4117;
  t4623 = t4621*t4120;
  t4628 = t4627*t4133;
  t4632 = t4631*t4151;
  t4642 = t4641*t4154;
  t4651 = t4650*t4170;
  t4659 = t4658*t4181;
  t4660 = -0.850685*t4183;
  t4661 = 0.069082*t4204;
  t4662 = -0.425556*t4217;
  t4663 = var1[2] + t4538 + t4542 + t4545 + t4550 + t4556 + t4562 + t4570 + t4574 + t4578 + t4583 + t4588 + t4593 + t4599 + t4603 + t4607 + t4612 + t4617 + t4623 + t4628 + t4632 + t4642 + t4651 + t4659 + t4660 + t4661 + t4662;
  t4665 = t4223*t3837*t4541;
  t4667 = -0.091*t3841*t4228;
  t4668 = -0.091*t4544*t4232;
  t4669 = t4549*t4236;
  t4670 = t4555*t4242;
  t4671 = t4561*t4244;
  t4672 = t4569*t4249;
  t4674 = t4573*t4256;
  t4675 = t4577*t4262;
  t4676 = t4582*t4267;
  t4679 = t4587*t4273;
  t4680 = t4592*t4275;
  t4681 = t4598*t4281;
  t4682 = t4602*t4289;
  t4683 = t4606*t4291;
  t4684 = t4611*t4297;
  t4685 = t4616*t4302;
  t4686 = t4621*t4304;
  t4687 = t4627*t4311;
  t4688 = t4631*t4324;
  t4689 = t4641*t4327;
  t4690 = t4650*t4333;
  t4691 = t4658*t4338;
  t4692 = -0.850685*t4340;
  t4693 = 0.069082*t4349;
  t4694 = -0.425556*t4354;
  t4695 = var1[0] + t4665 + t4667 + t4668 + t4669 + t4670 + t4671 + t4672 + t4674 + t4675 + t4676 + t4679 + t4680 + t4681 + t4682 + t4683 + t4684 + t4685 + t4686 + t4687 + t4688 + t4689 + t4690 + t4691 + t4692 + t4693 + t4694;
  t4697 = t3837*t4541*t4226;
  t4698 = -0.091*t3841*t4361;
  t4699 = -0.091*t4544*t4366;
  t4700 = t4549*t4368;
  t4701 = t4555*t4372;
  t4702 = t4561*t4374;
  t4703 = t4569*t4380;
  t4706 = t4573*t4385;
  t4707 = t4577*t4387;
  t4708 = t4582*t4395;
  t4709 = t4587*t4401;
  t4710 = t4592*t4403;
  t4711 = t4598*t4408;
  t4712 = t4602*t4414;
  t4713 = t4606*t4417;
  t4714 = t4611*t4426;
  t4715 = t4616*t4431;
  t4716 = t4621*t4433;
  t4717 = t4627*t4441;
  t4718 = t4631*t4449;
  t4719 = t4641*t4451;
  t4720 = t4650*t4456;
  t4721 = t4658*t4461;
  t4722 = -0.850685*t4463;
  t4723 = 0.069082*t4469;
  t4724 = -0.425556*t4474;
  t4725 = var1[1] + t4697 + t4698 + t4699 + t4700 + t4701 + t4702 + t4703 + t4706 + t4707 + t4708 + t4709 + t4710 + t4711 + t4712 + t4713 + t4714 + t4715 + t4716 + t4717 + t4718 + t4719 + t4720 + t4721 + t4722 + t4723 + t4724;
  p_output1[0]=-1.*t3454*t4221 + t3837*t4223*t4356 + t3837*t4226*t4476;
  p_output1[1]=t3837*t3849*t4221 + t4232*t4356 + t4366*t4476;
  p_output1[2]=t3837*t3839*t4221 + t4228*t4356 + t4361*t4476;
  p_output1[3]=0;
  p_output1[4]=-1.*t3454*t4489 + t3837*t4223*t4494 + t3837*t4226*t4498;
  p_output1[5]=t3837*t3849*t4489 + t4232*t4494 + t4366*t4498;
  p_output1[6]=t3837*t3839*t4489 + t4228*t4494 + t4361*t4498;
  p_output1[7]=0;
  p_output1[8]=-1.*t3454*t4512 + t3837*t4223*t4517 + t3837*t4226*t4523;
  p_output1[9]=t3837*t3849*t4512 + t4232*t4517 + t4366*t4523;
  p_output1[10]=t3837*t3839*t4512 + t4228*t4517 + t4361*t4523;
  p_output1[11]=0;
  p_output1[12]=-1.*t3454*t4663 + t3837*t4223*t4695 + t3837*t4226*t4725 - 1.*t3837*t4223*var1[0] - 1.*t3837*t4226*var1[1] + t3454*var1[2];
  p_output1[13]=t3837*t3849*t4663 + t4232*t4695 + t4366*t4725 - 1.*t4232*var1[0] - 1.*t4366*var1[1] - 1.*t3837*t3849*var1[2];
  p_output1[14]=t3837*t3839*t4663 + t4228*t4695 + t4361*t4725 - 1.*t4228*var1[0] - 1.*t4361*var1[1] - 1.*t3837*t3839*var1[2];
  p_output1[15]=1.;
}



#ifdef MATLAB_MEX_FILE

#include "mex.h"
/*
 * Main function
 */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
  size_t mrows, ncols;

  double *var1;
  double *p_output1;

  /*  Check for proper number of arguments.  */ 
  if( nrhs != 1)
    {
      mexErrMsgIdAndTxt("MATLAB:MShaped:invalidNumInputs", "One input(s) required (var1).");
    }
  else if( nlhs > 1)
    {
      mexErrMsgIdAndTxt("MATLAB:MShaped:maxlhs", "Too many output arguments.");
    }

  /*  The input must be a noncomplex double vector or scaler.  */
  mrows = mxGetM(prhs[0]);
  ncols = mxGetN(prhs[0]);
  if( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) ||
    ( !(mrows == 22 && ncols == 1) && 
      !(mrows == 1 && ncols == 22))) 
    {
      mexErrMsgIdAndTxt( "MATLAB:MShaped:inputNotRealVector", "var1 is wrong.");
    }

  /*  Assign pointers to each input.  */
  var1 = mxGetPr(prhs[0]);
   


   
  /*  Create matrices for return arguments.  */
  plhs[0] = mxCreateDoubleMatrix((mwSize) 4, (mwSize) 4, mxREAL);
  p_output1 = mxGetPr(plhs[0]);


  /* Call the calculation subroutine. */
  output1(p_output1,var1);


}

#else // MATLAB_MEX_FILE

#include "T_rightFoot.hh"

namespace SymExpression
{

void T_rightFoot_raw(double *p_output1, const double *var1)
{
  // Call Subroutines
  output1(p_output1, var1);

}

}

#endif // MATLAB_MEX_FILE

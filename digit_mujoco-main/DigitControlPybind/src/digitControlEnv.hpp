//
// Created by hyunyoungjung on 8/8/23.
//

#ifndef DIGIT_WS_DIGITVEC_H
#define DIGIT_WS_DIGITVEC_H
#include "omp.h"
#include "Yaml.hpp"
#include <Eigen/Dense>
#include "BasicEigenTypes.hpp"
#include "digit_controller.hpp"

class digitControlEnv {
    public:
    explicit digitControlEnv(double control_dt_) {
        control_dt = control_dt_;
    }

    ~digitControlEnv() {
        delete controller_;
    }
    void init(){
        controller_ = new digitcontroller::DigitController("/home/hjung331/mujoco/digit_IFM_mujoco/DigitControlPybind/src/include/digit_controller/src",control_dt);
    }

    void reset(Ref<EigenRowMajorMat> &digitStates){
        controller_->reset(digitStates.row(0));
    }

    void computeTorque(Ref<EigenRowMajorMat> &digitStates, Ref<EigenRowMajorMat> &torqueOut, Ref<EigenRowMajorMat> &velReferenceOut){
        controller_->computeTorque(digitStates.row(0), torqueOut.row(0), velReferenceOut.row(0));
    }
    void setUsrCommand(Ref<EigenRowMajorMat> &usrCommand){
        controller_->setUsrCommand(usrCommand.row(0));
    }
    double getPhaseVariable(){return controller_->getPhaseVariable();}
    int getDomain(){return controller_->getDomain();}
private:
    digitcontroller::DigitController * controller_;
    double control_dt;
};
#endif //DIGIT_WS_DIGITVEC_H

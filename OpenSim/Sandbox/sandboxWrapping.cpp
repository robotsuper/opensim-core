/* -------------------------------------------------------------------------- *
 * OpenSim: sandboxWrapping.cpp                                               *
 * -------------------------------------------------------------------------- *
 * Copyright (c) 2023 Stanford University and the Authors                     *
 *                                                                            *
 * Author(s): Nicholas Bianco                                                 *
 *                                                                            *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may    *
 * not use this file except in compliance with the License. You may obtain a  *
 * copy of the License at http://www.apache.org/licenses/LICENSE-2.0          *
 *                                                                            *
 * Unless required by applicable law or agreed to in writing, software        *
 * distributed under the License is distributed on an "AS IS" BASIS,          *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
 * See the License for the specific language governing permissions and        *
 * limitations under the License.                                             *
 * -------------------------------------------------------------------------- */

#include "OpenSim/Actuators/ModelFactory.h"
#include "OpenSim/Common/STOFileAdapter.h"

#include <OpenSim/Simulation/osimSimulation.h>

using namespace OpenSim;
using namespace SimTK;

class ImplicitSurfaceData : public Component {
OpenSim_DECLARE_CONCRETE_OBJECT(ImplicitSurfaceData, Component);

public:

    ImplicitSurfaceData() = default;

    const Vec3& getSurfacePosition() const { return p; }
    void setSurfacePosition(const Vec3& p) { this->p = p; }

    const Vec3& getOutwardNormal() const { return N; }
    void setOutwardNormal(const Vec3& N) { this->N = N; }

    const Real& getConstraintResidual() const { return f; }
    void setConstraintResidual(const Real& f) { this->f = f; }

    const Vec3& getConstraintFirstPartialDerivative() const { return df; }
    void setConstraintFirstPartialDerivative(const Vec3& df) { this->df = df; }

    const Vec6& getConstraintSecondPartialDerivative() const { return ddf; }
    void setConstraintSecondPartialDerivative(const Vec6& ddf) { this->ddf = ddf; }

    const Vec3& getConstraintGradient() const { return G; }
    void setConstraintGradient(const Vec3& G) { this->G = G; }

    const SymMat33& getConstraintHessian() const { return H; }
    void setConstraintHessian(const SymMat33& H) { this->H = H; }

    const Real& getGaussianCurvature() const { return K; }
    void setGaussianCurvature(const Real& K) { this->K = K; }

private:
    Vec3 p;
    Vec3 N;
    Real f = 0;
    Vec3 df;
    Vec6 ddf;
    Vec3 G;
    SymMat33 H;
    Real K = 0;
};


class ParametricSurfaceData : public Component {
OpenSim_DECLARE_CONCRETE_OBJECT(ParametricSurfaceData, Component);

public:
    ParametricSurfaceData() = default;

    void setU(const Real& u) { this->u = u; }
    const Real& getU() const { return u; }

    void setV(const Real& v) { this->v = v; }
    const Real& getV() const { return v; }

    void setX(const Vec3& x) { this->x = x; }
    const Vec3& getX() const { return x; }

    void setXU(const Vec3& xu) { this->xu = xu; }
    const Vec3& getXU() const { return xu; }

    void setXV(const Vec3& xv) { this->xv = xv; }
    const Vec3& getXV() const { return xv; }

    void setXUU(const Vec3& xuu) { this->xuu = xuu; }
    const Vec3& getXUU() const { return xuu; }

    void setXUV(const Vec3& xuv) { this->xuv = xuv; }
    const Vec3& getXUV() const { return xuv; }

    void setXVV(const Vec3& xvv) { this->xvv = xvv; }
    const Vec3& getXVV() const { return xvv; }

    void setOutwardNormal(const Vec3& N) { this->N = N; }
    const Vec3& getOutwardNormal() const { return N; }

    void setFirstFundamentalFormCoefficients(const Real& E, const Real& F, const Real& G) {
        this->E = E;
        this->F = F;
        this->G = G;
    }
    const Real& getE() const { return E; }
    const Real& getF() const { return F; }
    const Real& getG() const { return G; }

    void setSecondFundamentalFormCoefficients(const Real& e, const Real& f, const Real& g) {
        this->e = e;
        this->f = f;
        this->g = g;
    }
    const Real& gete() const { return e; }
    const Real& getf() const { return f; }
    const Real& getg() const { return g; }

    void setGaussianCurvature(const Real& K) { this->K = K; }
    const Real& getGaussianCurvature() const { return K; }

    void setMetricTensorDeterminant(const Real& detT) { this->detT = detT; }
    const Real& getMetricTensorDeterminant() const { return detT; }

private:
    Real u = 0, v = 0;			// Surface coordinates
    Vec3 x;						// Vector r(u, v) to point given by (u, v) with respect to and in coordinates of surface frame
    Vec3 xu, xv, xuu, xuv, xvv;	// Partial derivatives of surface parameterization
    Vec3 N;						// Outward normal vector

    Real E = 0, F = 0, G = 0;   // Coefficients of the first fundamental form
    Real e = 0, f = 0, g = 0;	// Coefficients of the second fundamental form

    Real K = 0;					// Gaussian curvature
    Real detT = 0;				// Determinant of metric tensor
};


class Surface : public Component {
OpenSim_DECLARE_ABSTRACT_OBJECT(Surface, Component);

public:
    OpenSim_DECLARE_SOCKET(frame, Frame,
        "The frame to which the surface is attached.");

    Surface() = default;

    void evaluateSurface(const Vec3& p) const {
        implicitSurfaceData.setSurfacePosition(p);

        evaluateImplicitEquation(p);
        evaluateSurfaceConstraintGradient();
        evaluateSurfaceNormalFromGradient();
        evaluateSurfaceConstraintHessian();
        evaluateGaussianCurvatureImplicitly();
    }

    void evaluateSurface(const Real& u, const Real& v) const {
        parametricSurfaceData.setU(u);
        parametricSurfaceData.setV(v);

        evaluateParametricEquation(u, v);

        evaluateFirstFundamentalForm();
        evaluateDeterminantOfMetricTensor();

        evaluateSecondFundamentalForm();
        evaluateGaussianCurvatureParametrically();
    }

    const ImplicitSurfaceData& getImplicitSurfaceData() const {
        return implicitSurfaceData;
    }

    ImplicitSurfaceData& updImplicitSurfaceData() const {
        return implicitSurfaceData;
    }

    const ParametricSurfaceData& getParametricSurfaceData() const {
        return parametricSurfaceData;
    }

    ParametricSurfaceData& updParametricSurfaceData() const {
        return parametricSurfaceData;
    }

protected:

    virtual void evaluateImplicitEquation(const SimTK::Vec3& p) const = 0;

    virtual void evaluateParametricEquation(const Real& u, const Real& v) const = 0;



private:

    void evaluateSurfaceConstraintGradient() const {
        implicitSurfaceData.setConstraintGradient(
            implicitSurfaceData.getConstraintFirstPartialDerivative());
    }

    void evaluateSurfaceNormalFromGradient() const {
        const auto& G = implicitSurfaceData.getConstraintGradient();
        implicitSurfaceData.setOutwardNormal(G / G.norm());
    }

    void evaluateSurfaceConstraintHessian() const {
        Vec6 ddf = implicitSurfaceData.getConstraintSecondPartialDerivative();
        SymMat33 H(ddf[0], ddf[1], ddf[2], ddf[3], ddf[4], ddf[5]);
        implicitSurfaceData.setConstraintHessian(H);
    }

    //		G * A * G^T
    // K = -------------- ,
    //        | G |^4
    //
    // where
    //
    //       [ Cofactor(fxx), Cofactor(fxy), Cofactor(fxz) ]   [ fyy*fzz-fyz*fzy, fyz*fzx-fyx*fzz, fyx*fzy-fyy*fzx ]
    //   A = [ Cofactor(fyx), Cofactor(fyy), Cofactor(fyz) ] = [ fxz*fzy-fxy*fzz, fxx*fzz-fxz*fzx, fxy*fzx-fxx*fzy ]
    // 	     [ Cofactor(fzx), Cofactor(fzy), Cofactor(fzz) ]   [ fxy*fyz-fxz*fyy, fyx*fxz-fxx*fyz, fxx*fyy-fxy*fyx ]
    //
    //  and by the Theorem of Schwarz it holds fab = fba, hence H' is a symmetric matrix.
    void evaluateGaussianCurvatureImplicitly() const {

        Vec6 ddf = implicitSurfaceData.getConstraintSecondPartialDerivative();
        const Real fxx = ddf[0];
        const Real fyy = ddf[1];
        const Real fzz = ddf[2];
        const Real fxy = ddf[3];
        const Real fxz = ddf[4];
        const Real fyz = ddf[5];

        // MoMatrix A(fyy*fzz-fyz*fyz, fyz*fxz-fxy*fzz, fxy*fyz-fyy*fxz,
        //            fxz*fyz-fxy*fzz, fxx*fzz-fxz*fxz, fxy*fxz-fxx*fyz,
        //            fxy*fyz-fxz*fyy, fxy*fxz-fxx*fyz, fxx*fyy-fxy*fxy);
        const SymMat33 A(fyy*fzz-fyz*fyz, fxx*fzz-fxz*fxz, fxx*fyy-fxy*fxy,
                         fxz*fyz-fxy*fzz, fxy*fyz-fxz*fyy, fxx*fyy-fxy*fxy);

        const auto& G = implicitSurfaceData.getConstraintGradient();
	    const Real normG = G.norm();

        const Real K = ~G * (A * G) / (normG * normG * normG * normG);
        implicitSurfaceData.setGaussianCurvature(K);
    }

    void evaluateFirstFundamentalForm() const {
        const Real E = ~parametricSurfaceData.getXU() * parametricSurfaceData.getXU();
        const Real F = ~parametricSurfaceData.getXU() * parametricSurfaceData.getXV();
        const Real G = ~parametricSurfaceData.getXV() * parametricSurfaceData.getXV();
        parametricSurfaceData.setFirstFundamentalFormCoefficients(E, F, G);
    }
    void evaluateDeterminantOfMetricTensor() const {
        const Real& E = parametricSurfaceData.getE();
        const Real& F = parametricSurfaceData.getF();
        const Real& G = parametricSurfaceData.getG();

        parametricSurfaceData.setMetricTensorDeterminant(E * G - F * F);
    }
    void evaluateSecondFundamentalForm() const {
        const Vec3& N = parametricSurfaceData.getOutwardNormal();
        const Vec3& xuu = parametricSurfaceData.getXUU();
        const Vec3& xuv = parametricSurfaceData.getXUV();
        const Vec3& xvv = parametricSurfaceData.getXVV();

        parametricSurfaceData.setSecondFundamentalFormCoefficients(
            ~xuu * N, ~xuv * N, ~xvv * N);
    }
    void evaluateGaussianCurvatureParametrically() const {
        const Real& e = parametricSurfaceData.gete();
        const Real& f = parametricSurfaceData.getf();
        const Real& g = parametricSurfaceData.getg();
        const Real& detT = parametricSurfaceData.getMetricTensorDeterminant();

        parametricSurfaceData.setGaussianCurvature((e * g - f * f) / detT);
    }

    mutable ImplicitSurfaceData implicitSurfaceData;
    mutable ParametricSurfaceData parametricSurfaceData;
};


class CylinderSurface : public Surface {
OpenSim_DECLARE_CONCRETE_OBJECT(CylinderSurface, Surface);

public:

    CylinderSurface(const Frame& frame,
               const Real& radius,
               const std::string& name) {
        connectSocket_frame(frame);
        setRadius(radius);
        setName(name);
    }

    void setRadius(const Real& radius) { this->radius = radius; }
    const Real& getRadius() const { return radius; }

    void evaluateImplicitEquation(const Vec3& p) const override {
        ImplicitSurfaceData& implicitSurfaceData =
            updImplicitSurfaceData();

        implicitSurfaceData.setSurfacePosition(p);

        implicitSurfaceData.setConstraintResidual(
            p[0] * p[0] + p[1] * p[1] - radius * radius);

        implicitSurfaceData.setConstraintFirstPartialDerivative(
            Vec3(2.0 * p[0], 2.0 * p[1], 0.0));

        implicitSurfaceData.setConstraintSecondPartialDerivative(
            Vec6(2.0, 2.0, 0.0, 0.0, 0.0, 0.0));
    }

    // Parametric cylinder
    // x(u, v) = radius * [ cos(u)  ]
    //                    [ sin(u)  ]
    //                    [ v       ]
    void evaluateParametricEquation(const Real& u, const Real& v) const override {
        const Real sinu = sin(u);
        const Real cosu = cos(u);
        ParametricSurfaceData& parametricSurfaceData =
            updParametricSurfaceData();

        parametricSurfaceData.setX({radius * cosu,
                                    radius * sinu,
                                    v});

        parametricSurfaceData.setXU({-radius * sinu,
                                      radius * cosu,
                                      0.0});

        parametricSurfaceData.setXV({0.0,
                                     0.0,
                                     1.0});

        Vec3 N =
           cross(parametricSurfaceData.getXU(), parametricSurfaceData.getXV());
        N = N.normalize();
        parametricSurfaceData.setOutwardNormal(N);


        parametricSurfaceData.setXUU({-radius * cosu,
                                      -radius * sinu,
                                      0.0});

        parametricSurfaceData.setXUV({0.0,
                                      0.0,
                                      0.0});

        parametricSurfaceData.setXVV({0.0,
                                      0.0,
                                      0.0});
    }

private:
    Real radius = 1.0;
};


class ImplicitGeodesicEquations : public Component {
OpenSim_DECLARE_CONCRETE_OBJECT(ImplicitGeodesicEquations, Component);

public:

    OpenSim_DECLARE_SOCKET(surface, Surface,
        "The surface to which the geodesic equations are attached.");

    ImplicitGeodesicEquations() = default;

    ImplicitGeodesicEquations(const Surface& surface,
                              const std::string& name) {
        connectSocket_surface(surface);
        setName(name);
    }

    const Surface& getSurface() const {
        return getSocket<Surface>("surface").getConnectee();
    }

    void evaluateStateSpaceGeodesicDifferentialEquations(const Vec3& p,
            const Vec3& dp, const Real& a, const Real& r,
            Vec3& ddp, Real& dda, Real& ddr) const {
        const auto& surface = getSurface();
        surface.evaluateSurface(p);

        const auto& data = surface.getImplicitSurfaceData();

        const Vec3& df = data.getConstraintFirstPartialDerivative();
        const Vec6& ddf = data.getConstraintSecondPartialDerivative();

        const Real& fx = df[0];
        const Real& fy = df[1];
        const Real& fz = df[2];

        const Real& fxx = ddf[0];
        const Real& fyy = ddf[1];
        const Real& fzz = ddf[2];
        const Real& fxy = ddf[3];
        const Real& fxz = ddf[4];
        const Real& fyz = ddf[5];

        const Real& dpx = dp[0];
        const Real& dpy = dp[1];
        const Real& dpz = dp[2];

        const Real& A = dpx*(fxx*dpx + fxy*dpy + fxz*dpz)
                      + dpy*(fxy*dpx + fyy*dpy + fyz*dpz)
                      + dpz*(fxz*dpx + fyz*dpy + fzz*dpz);

        const Real& D = fx*fx + fy*fy + fz*fz;

        // Geodesic implicit state space equations.
        ddp[0] = -fx * A / D;
        ddp[1] = -fy * A / D;
        ddp[2] = -fz * A / D;

        // Jacobi field equations.
        const auto& K = data.getGaussianCurvature();
        dda = -K * a;
        ddr = -K * r;
    }

protected:
    //--------------------------------------------------------------------------
    // COMPONENT INTERFACE
    //--------------------------------------------------------------------------
    /// @name Component interface
    /// @{
    void extendAddToSystem(MultibodySystem& system) const override {
        Super::extendAddToSystem(system);
        addStateVariable("px", Stage::Position);
        addStateVariable("py", Stage::Position);
        addStateVariable("pz", Stage::Position);
        addStateVariable("dpx", Stage::Position);
        addStateVariable("dpy", Stage::Position);
        addStateVariable("dpz", Stage::Position);
        addStateVariable("a", Stage::Position);
        addStateVariable("da", Stage::Position);
        addStateVariable("r", Stage::Position);
        addStateVariable("dr", Stage::Position);
    }
    void extendInitStateFromProperties(State& s) const override {
        setStateVariableValue(s, "px", 1.0);
        setStateVariableValue(s, "py", 0.0);
        setStateVariableValue(s, "pz", 0.0);
        setStateVariableValue(s, "dpx", 0.1);
        setStateVariableValue(s, "dpy", 0.1);
        setStateVariableValue(s, "dpz", 0.0);
        setStateVariableValue(s, "a", 1.0);
        setStateVariableValue(s, "da", 0.0);
        setStateVariableValue(s, "r", 0.0);
        setStateVariableValue(s, "dr", 1.0);
    }
    void computeStateVariableDerivatives(const State& s) const override {
        const Real& px = getStateVariableValue(s, "px");
        const Real& py = getStateVariableValue(s, "py");
        const Real& pz = getStateVariableValue(s, "pz");
        const Real& dpx = getStateVariableValue(s, "dpx");
        const Real& dpy = getStateVariableValue(s, "dpy");
        const Real& dpz = getStateVariableValue(s, "dpz");
        const Real& a = getStateVariableValue(s, "a");
        const Real& da = getStateVariableValue(s, "da");
        const Real& r = getStateVariableValue(s, "r");
        const Real& dr = getStateVariableValue(s, "dr");

        setStateVariableDerivativeValue(s, "px", dpx);
        setStateVariableDerivativeValue(s, "py", dpy);
        setStateVariableDerivativeValue(s, "pz", dpz);

        setStateVariableDerivativeValue(s, "a", da);
        setStateVariableDerivativeValue(s, "r", dr);

        Vec3 ddp;
        Real dda;
        Real ddr;
        const auto p = Vec3(px, py, pz);
        const auto dp = Vec3(dpx, dpy, dpz);
        evaluateStateSpaceGeodesicDifferentialEquations(p, dp, a, r,
            ddp, dda, ddr);

        setStateVariableDerivativeValue(s, "dpx", ddp[0]);
        setStateVariableDerivativeValue(s, "dpy", ddp[1]);
        setStateVariableDerivativeValue(s, "dpz", ddp[2]);
        setStateVariableDerivativeValue(s, "da", dda);
        setStateVariableDerivativeValue(s, "dr", ddr);
    }
    /// @}
};


class ParametricGeodesicEquations : public Component {
OpenSim_DECLARE_CONCRETE_OBJECT(ParametricGeodesicEquations, Component);

public:

    OpenSim_DECLARE_SOCKET(surface, Surface,
        "The surface to which the geodesic equations are attached.");

    ParametricGeodesicEquations() = default;

    ParametricGeodesicEquations(const Surface& surface,
                              const std::string& name) {
        connectSocket_surface(surface);
        setName(name);
    }

    void setPoints(const Vector_<Vec3>& points) {
        m_points = points;
    }

    const Surface& getSurface() const {
        return getSocket<Surface>("surface").getConnectee();
    }

    void evaluateStateSpaceGeodesicDifferentialEquations(const Real& u,
            const Real& v, const Real& du, const Real& dv,
            const Real& a, const Real& r,
            Real& ddu, Real& ddv, Real& dda, Real& ddr) const {
        const auto& surface = getSurface();
        surface.evaluateSurface(u, v);
        const auto& data = surface.getParametricSurfaceData();

        // First fundamental form
        const Real& E = data.getE();
        const Real& F = data.getF();
        const Real& G = data.getG();

        // Other coefficients
        const Real A = ~data.getXUU() * data.getXV();
        const Real B = ~data.getXU() * data.getXVV();
        const Real C = ~data.getXU() * data.getXUV();
        const Real D = ~data.getXUV() * data.getXV();

        const Real Eu = ~data.getXU() * data.getXUU();
        const Real Gv = ~data.getXV() * data.getXUV();

        // Determinant of metric tensor. For real surfaces, det(T) is always nonzero.
        const Real& detT = data.getMetricTensorDeterminant();

        assert(detT!=0);

        // Christoffel symbols
        const Real& T111 = (0.5 * Eu * G -       A * F ) / ( detT);
        const Real& T121 = (      C  * G -       D * F ) / ( detT);
        const Real& T221 = (      B  * G - 0.5 * F * Gv) / ( detT);

        const Real& T112 = (0.5 * Eu * F -       A * E ) / (-detT);
        const Real& T122 = (      C  * F -       D * E ) / (-detT);
        const Real& T222 = (      B  * F - 0.5 * E * Gv) / (-detT);

        // Geodesic parametric state space equations.
        ddu = -T111*du*du - 2.0*T121*du*dv - T221*dv*dv;
        ddv = -T112*du*du - 2.0*T122*du*dv - T222*dv*dv;

        // Jacobi field equations.
        const auto& K = data.getGaussianCurvature();
        dda = -K * a;
        ddr = -K * r;
    }

protected:
    //--------------------------------------------------------------------------
    // COMPONENT INTERFACE
    //--------------------------------------------------------------------------
    /// @name Component interface
    /// @{
    void extendAddToSystem(MultibodySystem& system) const override {
        Super::extendAddToSystem(system);
        addStateVariable("u", Stage::Position);
        addStateVariable("v", Stage::Position);
        addStateVariable("du", Stage::Position);
        addStateVariable("dv", Stage::Position);
        addStateVariable("a", Stage::Position);
        addStateVariable("da", Stage::Position);
        addStateVariable("r", Stage::Position);
        addStateVariable("dr", Stage::Position);
    }

    void extendInitStateFromProperties(State& s) const override {
        setStateVariableValue(s, "u", 2.1381);
        setStateVariableValue(s, "v", 0.0596);
        setStateVariableValue(s, "du", -0.9846);
        setStateVariableValue(s, "dv", -0.1750);
        setStateVariableValue(s, "a", 1.0);
        setStateVariableValue(s, "da", 0.0);
        setStateVariableValue(s, "r", 0.0);
        setStateVariableValue(s, "dr", 1.0);
    }

    void computeStateVariableDerivatives(const State& s) const override {
        const Real& u = getStateVariableValue(s, "u");
        const Real& v = getStateVariableValue(s, "v");
        const Real& du = getStateVariableValue(s, "du");
        const Real& dv = getStateVariableValue(s, "dv");
        const Real& a = getStateVariableValue(s, "a");
        const Real& da = getStateVariableValue(s, "da");
        const Real& r = getStateVariableValue(s, "r");
        const Real& dr = getStateVariableValue(s, "dr");

        setStateVariableDerivativeValue(s, "u", du);
        setStateVariableDerivativeValue(s, "v", dv);

        setStateVariableDerivativeValue(s, "a", da);
        setStateVariableDerivativeValue(s, "r", dr);

        Real ddu;
        Real ddv;
        Real dda;
        Real ddr;
        evaluateStateSpaceGeodesicDifferentialEquations(u, v, du, dv, a, r,
            ddu, ddv, dda, ddr);

        setStateVariableDerivativeValue(s, "du", ddu);
        setStateVariableDerivativeValue(s, "dv", ddv);
        setStateVariableDerivativeValue(s, "da", dda);
        setStateVariableDerivativeValue(s, "dr", ddr);
    }

    void generateDecorations(bool fixed, const ModelDisplayHints& hints,
        const State& s, Array_<DecorativeGeometry>& geometry) const override {
        Super::generateDecorations(fixed, hints, s, geometry);

        if (!m_points.size()) {
            return;
        }
        Vec3 lastPos(m_points[0]);
        for (int i = 1; i < (int)m_points.size(); ++i) {
            Vec3 pos(m_points[i]);
            // Line segments will be in ground frame
            geometry.push_back(DecorativeLine(lastPos, pos)
                .setLineThickness(4)
                .setColor({1.0, 0, 0}).setBodyId(0).setIndexOnBody(i));
            lastPos = pos;
        }

    }
    /// @}

private:
    Vector_<Vec3> m_points;
};


int main() {
    // Create a new OpenSim model.
    Model model;
    model.initSystem();

    Transform xform;

    auto* frame = new PhysicalOffsetFrame("frame", model.getGround(), xform);
    model.addComponent(frame);

    auto* cylinder = new CylinderSurface(*frame, 1.0, "cylinder");
    model.addComponent(cylinder);

    auto* wrapCyl = new WrapCylinder();
    wrapCyl->setFrame(*frame);
    wrapCyl->set_radius(1.0);
    wrapCyl->set_length(4.0);
    model.addComponent(wrapCyl);

    // Create a new OpenSim implicit geodesic equations.
    auto* geodesicEquations = new ParametricGeodesicEquations(*cylinder,
        "geodesic_equations");

    // Add the implicit geodesic equations to the model.
    model.addComponent(geodesicEquations);

    // Initialize the model.
    auto* statesReporter = new StatesTrajectoryReporter();
    statesReporter->setName("states_reporter");
    statesReporter->set_report_time_interval(0.01);
    model.addComponent(statesReporter);

    // Create a new OpenSim state.
    model.setUseVisualizer(true);
    auto state = model.initSystem();

    Manager manager(model);
    state.setTime(0.0);
    manager.initialize(state);
    manager.integrate(0.9772);

    auto states = statesReporter->getStates();
    TimeSeriesTable statesTable = states.exportToTable(model);
    STOFileAdapter::write(statesTable, "geodesicWrappingStates.sto");

    const auto& u = statesTable.getDependentColumn("/geodesic_equations/u");
    const auto& v = statesTable.getDependentColumn("/geodesic_equations/v");

    Vector_<Vec3> points((int)statesTable.getNumRows()+2, Vec3(0));
    Transform xform2;
    xform2.updR().setRotationToBodyFixedXYZ({-0.5*Pi, 0.0, 0});
    points[0] = xform2.p() + xform2.R() * Vec3(-5, -1, -2);
    for (int i = 1; i < (int)statesTable.getNumRows()+1; ++i) {
        cylinder->evaluateSurface(u[i-1], v[i-1]);
        const auto& x = cylinder->getParametricSurfaceData().getX();

        Vec3 xGlobal = xform.p() + xform.R() * x;
        points[i] = xGlobal;
    }
    points[points.size()-1] = xform2.p() + xform2.R() * Vec3(5, 1, -1);
    geodesicEquations->setPoints(points);

    model.getVisualizer().show(state);
}

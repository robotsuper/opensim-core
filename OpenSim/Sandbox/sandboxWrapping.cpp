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

// -----------------------------------------------------------------------------
// Common
// -----------------------------------------------------------------------------

class ImplicitSurfaceData : public Component {
OpenSim_DECLARE_CONCRETE_OBJECT(ImplicitSurfaceData, Component);

public:

    ImplicitSurfaceData() = default;

    const Vec3& getSurfacePosition() const { return p; }
    void setSurfacePosition(const Vec3& p) { this->p = p; }

    const Vec3& getOutwardNormal() const { return N; }
    void setOutwardNormal(const Vec3& N) { this->N = N; }

    const Real& getSurfaceConstraintResidual() const { return f; }
    void setSurfaceConstraintResidual(const Real& f) { this->f = f; }

    const Vec3& getConstraintFirstPartialDerivative() const { return df; }
    void setConstraintFirstPartialDerivative(const Vec3& df) { this->df = df; }

    const Vec6& getConstraintSecondPartialDerivative() const { return ddf; }
    void setConstraintSecondPartialDerivative(const Vec6& ddf) { this->ddf = ddf; }

    const Vec3& getSurfaceConstraintGradient() const { return G; }
    void setSurfaceConstraintGradient(const Vec3& G) { this->G = G; }

    const SymMat33& getSurfaceConstraintHessian() const { return H; }
    void setSurfaceConstraintHessian(const SymMat33& H) { this->H = H; }

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

// -----------------------------------------------------------------------------
// Simbody's 'Integrator'
// -----------------------------------------------------------------------------

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
        implicitSurfaceData.setSurfaceConstraintGradient(
            implicitSurfaceData.getConstraintFirstPartialDerivative());
    }

    void evaluateSurfaceNormalFromGradient() const {
        const auto& G = implicitSurfaceData.getSurfaceConstraintGradient();
        implicitSurfaceData.setOutwardNormal(G / G.norm());
    }

    void evaluateSurfaceConstraintHessian() const {
        Vec6 ddf = implicitSurfaceData.getConstraintSecondPartialDerivative();
        SymMat33 H(ddf[0], ddf[1], ddf[2], ddf[3], ddf[4], ddf[5]);
        implicitSurfaceData.setSurfaceConstraintHessian(H);
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

        const auto& G = implicitSurfaceData.getSurfaceConstraintGradient();
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

        implicitSurfaceData.setSurfaceConstraintResidual(
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

    void setPoints(const Vector_<Vec3>& points) {
        m_points = points;
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
private:
    Vector_<Vec3> m_points;
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

// -----------------------------------------------------------------------------
// GeodesicIntegrator
// -----------------------------------------------------------------------------

class GeodesicSurfaceImpl {

public:
    GeodesicSurfaceImpl() = default;
    virtual ~GeodesicSurfaceImpl() = default;

    void evaluateSurface(const Vec3& p) const {
        implicitSurfaceData.setSurfacePosition(p);

        evaluateImplicitEquation(p);
        evaluateSurfaceConstraintGradient();
        evaluateSurfaceNormalFromGradient();
        evaluateSurfaceConstraintHessian();
        evaluateGaussianCurvatureImplicitly();
    }

    const ImplicitSurfaceData& getImplicitSurfaceData() const {
        return implicitSurfaceData;
    }

    ImplicitSurfaceData& updImplicitSurfaceData() const {
        return implicitSurfaceData;
    }

protected:
    virtual void evaluateImplicitEquation(const SimTK::Vec3& p) const = 0;

private:

    void evaluateSurfaceConstraintGradient() const {
        implicitSurfaceData.setSurfaceConstraintGradient(
                implicitSurfaceData.getConstraintFirstPartialDerivative());
    }

    void evaluateSurfaceNormalFromGradient() const {
        const auto& G = implicitSurfaceData.getSurfaceConstraintGradient();
        implicitSurfaceData.setOutwardNormal(G / G.norm());
    }

    void evaluateSurfaceConstraintHessian() const {
        Vec6 ddf = implicitSurfaceData.getConstraintSecondPartialDerivative();
        SymMat33 H(ddf[0], ddf[1], ddf[2], ddf[3], ddf[4], ddf[5]);
        implicitSurfaceData.setSurfaceConstraintHessian(H);
    }

    void evaluateGaussianCurvatureImplicitly() const {

        Vec6 ddf = implicitSurfaceData.getConstraintSecondPartialDerivative();
        const Real fxx = ddf[0];
        const Real fyy = ddf[1];
        const Real fzz = ddf[2];
        const Real fxy = ddf[3];
        const Real fxz = ddf[4];
        const Real fyz = ddf[5];

        const SymMat33 A(fyy*fzz-fyz*fyz, fxx*fzz-fxz*fxz, fxx*fyy-fxy*fxy,
                fxz*fyz-fxy*fzz, fxy*fyz-fxz*fyy, fxx*fyy-fxy*fxy);

        const auto& G = implicitSurfaceData.getSurfaceConstraintGradient();
        const Real normG = G.norm();

        const Real K = ~G * (A * G) / (normG * normG * normG * normG);
        implicitSurfaceData.setGaussianCurvature(K);
    }


    mutable ImplicitSurfaceData implicitSurfaceData;
};



class CylinderGeodesicSurfaceImpl : public GeodesicSurfaceImpl {

public:

    CylinderGeodesicSurfaceImpl(const Real& radius) {
        setRadius(radius);
    }

    void setRadius(const Real& radius) { this->radius = radius; }
    const Real& getRadius() const { return radius; }

    void evaluateImplicitEquation(const Vec3& p) const override {
        ImplicitSurfaceData& implicitSurfaceData =
                updImplicitSurfaceData();

        implicitSurfaceData.setSurfacePosition(p);

        implicitSurfaceData.setSurfaceConstraintResidual(
                p[0] * p[0] + p[1] * p[1] - radius * radius);

        implicitSurfaceData.setConstraintFirstPartialDerivative(
                Vec3(2.0 * p[0], 2.0 * p[1], 0.0));

        implicitSurfaceData.setConstraintSecondPartialDerivative(
                Vec6(2.0, 2.0, 0.0, 0.0, 0.0, 0.0));
    }


private:
    Real radius = 1.0;
};

// https://github.com/simbody/simbody/blob/0f0dcb9b214b2e7def038d50261c99f6eb4df166/SimTKmath/Geometry/src/GeodesicEquations.h
// https://github.com/simbody/simbody/blob/0f0dcb9b214b2e7def038d50261c99f6eb4df166/SimTKmath/Geometry/src/Geodesic.cpp
class GeodesicEquationsImplicitSurface {
public:
    // state y = q | u = px py pz jr jt | vx vy vz jrd jtd
    // NQ, NC are required by the GeodesicIntegrator
    enum { D = 3,       // 3 coordinates for a point on an implicit surface.
        NJ = 2,      // Number of Jacobi field equations.
        NQ = D+NJ,   // Number of 2nd order equations.
        N  = 2*NQ,   // Number of differential equations.
        NC = 3 };    // 3 constraints: point on surface, point velocity
                     // along surface, unit velocity

    GeodesicEquationsImplicitSurface(const GeodesicSurfaceImpl& surface) : _surface(surface) {}

    // Calculate state derivatives ydot given time and state y.
    void calcDerivs(Real t, const Vec<N>& y, Vec<N>& ydot) const {

        const Vec3& p = getP(y);        // rename state variables
        const Vec3& v = getV(y);
        const Real& jr = getJRot(y);
        const Real& jt = getJTrans(y);

        // Evaluate the surface at p.
        // TODO inefficent: should only calculate what we need.
        _surface.evaluateSurface(p);
        const auto& data = _surface.getImplicitSurfaceData();

        const Vec3  g = data.getSurfaceConstraintGradient();
        const SymMat33 H = data.getSurfaceConstraintHessian();
        Real Kg = data.getGaussianCurvature();

        const Real Gdotv = ~v*(H*v);
        const Real L = Gdotv/(~g*g);    // Lagrange multiplier

        // We have qdot = u; that part is easy.
        updQ(ydot) = getU(y);

        // These together are the udots.
        Vec3& a     = updV(ydot);          // d/dt v
        Real& jrdd  = updJRotDot(ydot);    // d/dt jdr
        Real& jtdd  = updJTransDot(ydot);  // d/dt jdt

        a    = -L*g;
        jrdd = -Kg*jr;
        jtdd = -Kg*jt;
    }

    // Calculate amount by which the given time and state y violate the
    // constraints and return the constraint errors in cerr.
    void calcConstraintErrors(Real t, const Vec<N>& y, Vec<NC>& cerr) const {
        const Vec3& p = getP(y);
        const Vec3& v = getV(y);

        // Evaluate the surface at p.
        // TODO inefficent: should only calculate what we need.
        _surface.evaluateSurface(p);
        const auto& data = _surface.getImplicitSurfaceData();

        // This is the perr() equation that says the point must be on the surface.
        cerr[0] = data.getSurfaceConstraintResidual();
        // These are the two verr() equations. The first is the derivative of
        // the above point-on-surface holonomic constraint above. The second is
        // a nonholonomic velocity constraint restricting the velocity along
        // the curve to be 1.
        std::cout << "data.getSurfaceConstraintGradient: " << data.getSurfaceConstraintGradient() << std::endl;
        cerr[1] = ~data.getSurfaceConstraintGradient()*v;
        cerr[2] = v.norm() - 1;

    }

    // Given a time and state y, ensure that the state satisfies the constraints
    // to within the indicated absolute tolerance, by performing the shortest
    // (i.e. least squares) projection of the state back to the constraint
    // manifold. Return false if the desired tolerance cannot be achieved.
    // Otherwise (true return), a subsequent call to calcConstraintErrors()
    // would return each |cerr[i]|<=consTol.
    bool projectIfNeeded(Real consTol, Real t, Vec<N>& y) const {
        const int MaxIter = 10;         // should take *far* fewer
        const Real OvershootFac = Real(0.1);  // try to do better than consTol

        const Real tryTol = consTol * OvershootFac;
        Vec3& p = updP(y); // aliases for the state variables
        Vec3& v = updV(y);

        // Evaluate the surface at p.
        // TODO inefficent: should only calculate what we need.
        _surface.evaluateSurface(p);
        const auto& data = _surface.getImplicitSurfaceData();

        // Fix the position constraint first. This is a Newton iteration
        // that modifies only the point location to make sure it remains on
        // the surface. No position projection is done if we're already at
        // tryTol, which is a little tighter than the requested consTol.

        // NOTE: (sherm) I don't think this is exactly the right projection.
        // Here we project down the gradient, but the final result won't
        // be exactly the nearest point on the surface if the gradient changes
        // direction on the way down. For correcting small errors this is
        // probably completely irrelevant since the starting and final gradient
        // directions will be the same.

        Real perr, ptolAchieved;
        int piters=0;
        while (true) {
            perr = data.getSurfaceConstraintResidual();
            ptolAchieved = std::abs(perr);
            std::cout << "ptolAchieved: " << ptolAchieved << std::endl;
            if (ptolAchieved <= tryTol || piters==MaxIter)
                break;

            ++piters;
            // We want a least squares solution dp to ~g*dp=perr which we
            // get using the pseudoinverse: dp=pinv(~g)*perr, where
            // pinv(~g) = g*inv(~g*g).
            const Vec3 g = data.getSurfaceConstraintGradient();
            const Vec3 pinvgt = g/(~g*g);
            const Vec3 dp = pinvgt*perr;

            p -= dp; // updates the state
        }


        // Now the velocities. There are two velocity constraints that have
        // to be satisfied simultaneously. They are (1) the time derivative of
        // the perr equation which we just solved, and (2) the requirement that
        // the velocity magnitude be 1. So verr=~[ ~g*v, |v|-1 ]. You might
        // think these need to be solved simultaneously to find the least
        // squares dv, but dv can be determined by two orthogonal projections.
        // The allowable velocity vectors form a unit circle whose normal is
        // in the gradient direction. The least squares dv is the shortest
        // vector from the end point of v to that circle. To find the closest
        // point on the unit circle, first project the vector v onto the
        // circle's plane by the shortest path (remove the normal component).
        // Then stretch the result to unit length.
        // First we solve the linear least squares problem ~g*(v+dv0)=0 for
        // dv0, and set v0=v+dv0. Then set vfinal = v0/|v0|, giving dv=vfinal-v.

        // We're going to project velocities unconditionally because we
        // would have to evaluate the constraint anyway to see if it is
        // violated and that is most of the computation we need to fix it.

        const Vec3 g = data.getSurfaceConstraintGradient();
        const Vec3 pinvgt = g/(~g*g);
        const Real perrdot = ~g*v;

        const Vec3 dv0 = pinvgt*perrdot;
        const Vec3 v0 = v - dv0;    // fix direction
        v = v0/v0.norm();           // fix length; updates state

        const bool success = (ptolAchieved <= consTol);
        return success;
    }

    // Utility routine for filling in the initial state given a starting
    // point and direction. Note that there is no guarantee that the resulting
    // state satisfies the constraints.
    static Vec<N> getInitialState(const Vec3& P, const UnitVec3& tP) {
        Vec<N> y;
        updP(y)      = P;   updV(y)         = tP.asVec3();
        updJRot(y)   = 0;   updJRotDot(y)   = 1;
        updJTrans(y) = 1;   updJTransDot(y) = 0;
        return y;
    }

    // These define the meanings of the state variables & derivatives.
    static const Vec<NQ>& getQ(const Vec<N>& y) {return Vec<NQ>::getAs(&y[0]);}
    static Vec<NQ>& updQ(Vec<N>& y) {return Vec<NQ>::updAs(&y[0]);}

    static const Vec<NQ>& getU(const Vec<N>& y) {return Vec<NQ>::getAs(&y[NQ]);}
    static Vec<NQ>& updU(Vec<N>& y) {return Vec<NQ>::updAs(&y[NQ]);}


    // Extract the point location from a full state y.
    static const Vec3& getP(const Vec<N>& y) {return Vec3::getAs(&y[0]);}
    static Vec3& updP(Vec<N>& y) {return Vec3::updAs(&y[0]);}
    // Extract the point velocity from a full state y.
    static const Vec3& getV(const Vec<N>& y) {return Vec3::getAs(&y[NQ]);}
    static Vec3& updV(Vec<N>& y) {return Vec3::updAs(&y[NQ]);}

    // Extract the value of the rotational Jacobi field from a state y.
    static const Real& getJRot(const Vec<N>& y) {return y[D];}
    static Real& updJRot(Vec<N>& y) {return y[D];}
    static const Real& getJRotDot(const Vec<N>& y) {return y[NQ+D];}
    static Real& updJRotDot(Vec<N>& y) {return y[NQ+D];}
    // Extract the value of the translational Jacobi field from a state y.
    static const Real& getJTrans(const Vec<N>& y) {return y[D+1];}
    static Real& updJTrans(Vec<N>& y) {return y[D+1];}
    static const Real& getJTransDot(const Vec<N>& y) {return y[NQ+D+1];}
    static Real& updJTransDot(Vec<N>& y) {return y[NQ+D+1];}

private:

    const GeodesicSurfaceImpl& _surface;

};


int main() {

    // Create a new OpenSim model.
    Model model;
    model.initSystem();

    Transform xform;

    auto* frame =
            new PhysicalOffsetFrame("frame", model.getGround(), xform);
    model.addComponent(frame);

    auto* cylinder = new CylinderSurface(*frame, 1.0, "cylinder");
    model.addComponent(cylinder);

    auto* wrapCyl = new WrapCylinder();
    wrapCyl->setFrame(*frame);
    wrapCyl->set_radius(1.0);
    wrapCyl->set_length(4.0);
    model.addComponent(wrapCyl);

    // Integration using the ModelComponent interface
    // (i.e., Simbody's Integrator)
    if (false) {
        // Create a new OpenSim implicit geodesic equations.
        auto* geodesicEquations = new ParametricGeodesicEquations(
                *cylinder, "geodesic_equations");

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

        Vector_<Vec3> points((int)statesTable.getNumRows() + 2, Vec3(0));
        Transform xform2;
        xform2.updR().setRotationToBodyFixedXYZ({-0.5 * Pi, 0.0, 0});
        points[0] = xform2.p() + xform2.R() * Vec3(-5, -1, -2);
        for (int i = 1; i < (int)statesTable.getNumRows() + 1; ++i) {
            cylinder->evaluateSurface(u[i - 1], v[i - 1]);
            const auto& x = cylinder->getParametricSurfaceData().getX();

            Vec3 xGlobal = xform.p() + xform.R() * x;
            points[i] = xGlobal;
        }
        points[points.size() - 1] = xform2.p() + xform2.R() * Vec3(5, 1, -1);
        geodesicEquations->setPoints(points);

        model.getVisualizer().show(state);
    }

    // Integration using GeodesicIntegrator
    if (true) {
        auto* cylinderGeodesicSurfaceImpl = new CylinderGeodesicSurfaceImpl(1.0);
        std::unique_ptr<GeodesicSurfaceImpl> cylinderGeodesicSurfaceImplPtr(
                cylinderGeodesicSurfaceImpl);

        GeodesicEquationsImplicitSurface geodesicEquations(*cylinderGeodesicSurfaceImpl);

        // TODO constraint projections fail for values smaller than 1e-1 (for accuracy 1e-10)
        Real accuracy = 1e-10;
        Real consTol = 1e-1;
        GeodesicIntegrator<GeodesicEquationsImplicitSurface> integrator(geodesicEquations, accuracy, consTol);

        // Set the initial and final arc length.
        Real s0 = 0.0;
        Real sf = 0.9772;

        // Set the initial state.
        // Initial values based on Andreas' matlab code for a cylinder at a
        // hard-coded orientation and position with specified origin and
        // insertion points.
        Vec3 origin(-5, -1, -2);
        Vec3 insertion(5, 1, -1);
        Vec3 p0(-0.5373, 0.8434, 0.0596);
        UnitVec3 v0(0.8304, 0.5291, -0.1750);
        Vec<10> y0 = GeodesicEquationsImplicitSurface::getInitialState(p0, v0);

        // Initialize; will project constraints if necessary.
        integrator.initialize(s0, y0);

        // Integrate to final arc length, getting output every completed step.
        std::vector<Vec3> pointsTemp;
        while (true) {
            pointsTemp.push_back(integrator.getY().getSubVec<3>(0));
            std::cout << "t=" << integrator.getTime() << " y=" << integrator.getY() << "\n";
            std::cout << "\n";
            if (integrator.getTime() == sf)
                break;
            integrator.takeOneStep(sf);
        }

        // Only for visualization
        // Must run in debug, otherwise the visualization window will close immediately
        auto* eqns = new ImplicitGeodesicEquations(
                *cylinder, "geodesic_equations");
        model.addComponent(eqns);
        model.setUseVisualizer(true);
        auto state = model.initSystem();

        Vector_<Vec3> points((int)pointsTemp.size()+2, Vec3(0));
        // These transformations are just to account for the differences in
        // conventions between the Matlab code and OpenSim.
        Transform xform2;
        xform2.updR().setRotationToBodyFixedXYZ({-0.5 * Pi, 0.0, 0});
        points[0] = xform2.p() + xform2.R() * origin;
        for (int i = 1; i < (int)pointsTemp.size() + 1; ++i) {
            const auto& x = pointsTemp[i - 1];

            Vec3 xGlobal = xform.p() + xform.R() * x;
            points[i] = xGlobal;
        }
        points[points.size() - 1] = xform2.p() + xform2.R() * insertion;
        eqns->setPoints(points);

        model.getVisualizer().show(state);
    }

    return EXIT_SUCCESS;
}

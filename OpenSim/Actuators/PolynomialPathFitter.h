#ifndef OPENSIM_POLYNOMIALPATHFITTER_H
#define OPENSIM_POLYNOMIALPATHFITTER_H
/* -------------------------------------------------------------------------- *
 *                    OpenSim:  PolynomialPathFitter.h                        *
 * -------------------------------------------------------------------------- *
 * The OpenSim API is a toolkit for musculoskeletal modeling and simulation.  *
 * See http://opensim.stanford.edu and the NOTICE file for more information.  *
 * OpenSim is developed at Stanford University and supported by the US        *
 * National Institutes of Health (U54 GM072970, R24 HD065690) and by DARPA    *
 * through the Warrior Web program.                                           *
 *                                                                            *
 * Copyright (c) 2005-2023 Stanford University and the Authors                *
 * Author(s): Nicholas Bianco                                                 *
 *                                                                            *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may    *
 * not use this file except in compliance with the License. You may obtain a  *
 * copy of the License at http://www.apache.org/licenses/LICENSE-2.0.         *
 *                                                                            *
 * Unless required by applicable law or agreed to in writing, software        *
 * distributed under the License is distributed on an "AS IS" BASIS,          *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
 * See the License for the specific language governing permissions and        *
 * limitations under the License.                                             *
 * -------------------------------------------------------------------------- */

#include <OpenSim/Simulation/TableProcessor.h>
#include <OpenSim/Actuators/ModelProcessor.h>
#include <OpenSim/Simulation/Model/FunctionBasedPath.h>

namespace OpenSim {

/**
 * A helper class for specifying the minimum and maximum bounds for the
 * coordinate at `coordinate_path` during path fitting. The bounds are
 * specified as a `SimTK::Vec2` in the property `bounds`, where the first
 * element is the minimum bound and the second element is the maximum bound.
 */
class OSIMACTUATORS_API PolynomialPathFitterBounds : public Object {
    OpenSim_DECLARE_CONCRETE_OBJECT(PolynomialPathFitterBounds, Object);

public:
    OpenSim_DECLARE_PROPERTY(coordinate_path, std::string,
            "The path to the bounded coordinate in the model.")
    OpenSim_DECLARE_PROPERTY(bounds, SimTK::Vec2,
            "The bounds for the coordinate. The first element is the minimum "
            "bound and the second element is the maximum bound.")
    PolynomialPathFitterBounds();
    PolynomialPathFitterBounds(
            const std::string& coordinatePath, const SimTK::Vec2& bounds);
private:
    void constructProperties();
};

/**
 * A class for fitting a set of `FunctionBasedPath`s to paths in an OpenSim
 * model.
 */
class OSIMACTUATORS_API PolynomialPathFitter : public Object {
    OpenSim_DECLARE_CONCRETE_OBJECT(PolynomialPathFitter, Object);

public:
    // CONSTRUCTION AND DESTRUCTION
    PolynomialPathFitter();
    ~PolynomialPathFitter() noexcept override;
    PolynomialPathFitter(const PolynomialPathFitter&);
    PolynomialPathFitter(PolynomialPathFitter&&);
    PolynomialPathFitter& operator=(const PolynomialPathFitter&);
    PolynomialPathFitter& operator=(PolynomialPathFitter&&);

    // MAIN INPUTS
    /**
     * The model containing geometry-based path objects to which
     * polynomial-based path objects will be fitted.
     *
     * The model should be provided using a `ModelProcessor` object. We expect
     * the model to contain at least one path object derived from `AbstractPath`
     * and does not already contain any `FunctionBasedPath` objects. The bounds
     * for clamped coordinates are obeyed during the fitting process. Locked
     * coordinates are unlocked if data is provided for them, or replaced with
     * WeldJoints if no data is provided for them.
     */
    void setModel(ModelProcessor model);

    /**
     * The reference trajectory used to sample coordinate values for path
     * fitting.
     *
     * The reference trajectory should be provided using a `TableProcessor`
     * object. The reference trajectory must contain coordinate values for all
     * `Coordinate`s in the model. We assumed that the coordinate values meet
     * all the kinematic constraints in the model, except for
     * `CoordinateCouplerConstraint`s, since we automatically update the
     * coordinate trajectory to satisfy these constraints. The `TimeSeriesTable`
     * must contain the "inDegrees" metadata flag; the coordinate values are
     * automatically converted to radians if this flag is set to true.
     */
    void setCoordinateValues(TableProcessor coordinateValues);

    // RUN PATH FITTING
    void run();

    // SETTINGS
    /**
     * The directory to which the path fitting results are written.
     *
     * If the path fitting is successful, the fitted paths are written as a
     * `Set` of `FunctionBasedPath` objects (with path length functions defined
     * using `MultivariatePolynomialFunction` objects) to an XML file. Files
     * containing the modified coordinate values, sampled coordinate values,
     * path lengths, and moment arms for both the original and fitted paths are
     * also written to the output directory.
     *
     * @note By default, results are written to the current working directory.
     */
    void setOutputDirectory(std::string directory);
    std::string getOutputDirectory() const;

    /**
     * The moment arm threshold value that determines whether or not a path
     * depends on a model coordinate. In other words, the moment arm of a path
     * with respect to a particular coordinate must be greater than this value
     * to be included during path fitting.
     */
    void setMomentArmThreshold(double threshold);
    /// @copydoc setMomentArmThreshold()
    double getMomentArmThreshold() const;

    /**
     * The minimum order of the polynomial used to fit each path. The order of
     * a polynomial is the highest power of the independent variable(s) in the
     * polynomial.
     */
    void setMinimumPolynomialOrder(int order);
    /// @copydoc setMinimumPolynomialOrder()
    int getMinimumPolynomialOrder() const;

    /**
     * The maximum order of the polynomial used to fit each path. The order of
     * a polynomial is the highest power of the independent variable(s) in the
     * polynomial.
     */
    void setMaximumPolynomialOrder(int order);
    /// @copydoc setMaximumPolynomialOrder()
    int getMaximumPolynomialOrder() const;

    /**
     * The global bounds (in degrees) that determine the minimum and maximum
     * coordinate value samples at each time point.
     *
     * The bounds are specified as a `SimTK::Vec2`, where the first element is
     * the minimum bound and the second element is the maximum bound. The .
     * maximum sample value at a particular time point is the nominal coordinate
     * value plus the maximum bound, and the minimum sample value is the
     * nominal coordinate value minus the minimum bound.
     *
     * @note The default global bounds are set to [-10, 10] degrees.
     *
     * @note To override the default global bounds for a specific coordinate,
     *       use the `appendCoordinateSamplingBounds()` method.
     */
    void setGlobalCoordinateSamplingBounds(SimTK::Vec2 bounds);
    /// @copydoc setGlobalCoordinateSamplingBounds()
    SimTK::Vec2 getGlobalCoordinateSamplingBounds() const;

    /**
     * The bounds (in degrees) that determine the minimum and maximum coordinate
     * value samples at each time point for the coordinate at `coordinatePath`.
     *
     * The bounds are specified as a `SimTK::Vec2`, where the first element is
     * the minimum bound and the second element is the maximum bound. The
     * maximum sample value at a particular time point is the nominal coordinate
     * value plus the maximum bound, and the minimum sample value is the
     * nominal coordinate value minus the minimum bound. This overrides the
     * global bounds set by `setGlobalCoordinateSamplingBounds()` for this
     * coordinate.
     */
     void appendCoordinateSamplingBounds(
            const std::string& coordinatePath, const SimTK::Vec2& bounds);

    /**
     * The tolerance on the root-mean-square (RMS) error (in meters) between the
     * moment arms computed from an original model path and a fitted
     * polynomial-based path, which is used to determine the order of the
     * polynomial used in the fitted path.
     *
     * The moment arm RMS error must be less than the tolerance for the
     * polynomial order to be accepted. If the RMS error is greater than the
     * tolerance, the polynomial order is increased by one and the path is
     * refitted. This process is repeated until the RMS error is less than the
     * tolerance or the maximum polynomial order is reached.
     *
     * @note The default moment arm tolerance is set to 1e-4 meters.
     * @note The path length RMS error must also be less than the path length
     *       tolerance for the polynomial order to be accepted (see
     *       `setPathLengthTolerance`).
     */
    void setMomentArmTolerance(double tolerance);
    /// @copydoc setMomentArmTolerance()
    double getMomentArmTolerance() const;

    /**
     * The tolerance on the root-mean-square (RMS) error (in meters) between the
     * path lengths computed from an original model path and a fitted
     * polynomial-based path, which is used to determine the order of the
     * polynomial used in the fitted path.
     *
     * The path length RMS error must be less than the tolerance for the
     * polynomial order to be accepted. If the RMS error is greater than the
     * tolerance, the polynomial order is increased by one and the path is
     * refitted. This process is repeated until the RMS error is less than the
     * tolerance or the maximum polynomial order is reached.
     *
     * @note The default path length tolerance is set to 1e-4 meters.
     * @note The moment arm RMS error must also be less than the moment arm
     *       tolerance for the polynomial order to be accepted (see
     *      `setMomentArmTolerance`).
     */
    void setPathLengthTolerance(double tolerance);
    /// @copydoc setPathLengthTolerance()
    double getPathLengthTolerance() const;

    /**
     * The number of samples taken per time frame in the coordinate values table
     * used to fit each path.
     *
     * @note The default number of samples per frame is set to 25.
     */
    void setNumSamplesPerFrame(int numSamples);
    /// @copydoc setNumSamplesPerFrame()
    int getNumSamplesPerFrame() const;

    /**
     * The number of threads used to parallelize the path fitting process.
     *
     * This setting is used to divide the coordinate sampling, path length and
     * moment arm computations, and path fitting across multiple threads. The
     * number of threads must be greater than zero.
     *
     * @note The default number of threads is set to two fewer than the number
     *       of available hardware threads.
     */
    void setParallel(int numThreads);
    /// @copydoc setParallel()
    int getParallel() const;

    /**
     * The Latin hypercube sampling algorithm used to sample coordinate values
     * for path fitting.
     *
     * The Latin hypercube sampling algorithm is used to sample coordinate
     * values for path fitting. The algorithm can be set to either "random" or
     * "ESEA", which stands for the enhanced stochastic evolutionary algorithm
     * developed by Jin et al. 2005 (see class `LatinHypercubeDesign` for more
     * details). The "random" algorithm is used by default, and "ESEA" can be
     * used to improve the quality of the sampling at the expense of higher
     * computational cost. For most applications, the "random" algorithm is
     * likely sufficient.
     */
    void setLatinHypercubeAlgorithm(std::string algorithm);
    /// @copydoc setLatinHypercubeAlgorithm()
    std::string getLatinHypercubeAlgorithm() const;

    // HELPER FUNCTIONS
    static void evaluateFittedPaths(Model model,
            TableProcessor trajectory,
            const std::string& functionBasedPathsFileName);

private:
    // PROPERTIES
    OpenSim_DECLARE_PROPERTY(model, ModelProcessor,
            "The model containing geometry-based path objects to which "
            "polynomial-based path objects will be fitted.");
    OpenSim_DECLARE_PROPERTY(coordinate_values, TableProcessor,
            "The reference trajectory used to sample coordinate values for "
            "path fitting.");
    OpenSim_DECLARE_PROPERTY(output_directory, std::string,
            "The directory to which the path fitting results are written.");
    OpenSim_DECLARE_PROPERTY(moment_arm_threshold, double,
            "The moment arm threshold value that determines whether or not a "
            "path depends on a model coordinate. In other words, the moment "
            "arm of a path with respect to a coordinate must be greater than "
            "this value to be included during path fitting.");
    OpenSim_DECLARE_PROPERTY(minimum_polynomial_order, int,
            "The minimum order of the polynomial used to fit each path. The "
            "order of a polynomial is the highest power of the independent "
            "variable(s) in the polynomial.");
    OpenSim_DECLARE_PROPERTY(maximum_polynomial_order, int,
            "The maximum order of the polynomial used to fit each path. The "
            "order of a polynomial is the highest power of the independent "
            "variable(s) in the polynomial.");
    OpenSim_DECLARE_PROPERTY(global_coordinate_sampling_bounds, SimTK::Vec2,
            "The global bounds (in degrees) that determine the minimum and "
            "maximum coordinate value samples at each time point.");
    OpenSim_DECLARE_LIST_PROPERTY(
            coordinate_sampling_bounds, PolynomialPathFitterBounds,
            "The bounds (in degrees) that determine the minimum and maximum "
            "coordinate value samples at each time point for specific "
            "coordinates. These bounds override the default coordinate "
            "sampling bounds.");
    OpenSim_DECLARE_PROPERTY(moment_arm_tolerance, double,
            "The tolerance on the root-mean-square (RMS) error (in meters) "
            "between the moment arms computed from an original model path and "
            "a fitted polynomial-based path, which is used to determine the "
            "order of the polynomial used in the fitted path (default: 1e-4).");
    OpenSim_DECLARE_PROPERTY(path_length_tolerance, double,
            "The tolerance on the root-mean-square (RMS) error (in meters) "
            "between the path lengths computed from an original model path and "
            "a fitted polynomial-based path, which is used to determine the "
            "order of the polynomial used in the fitted path (default: 1e-4).");
    OpenSim_DECLARE_PROPERTY(num_samples_per_frame, int,
            "The number of samples taken per time frame in the coordinate "
            "values table used to fit each path (default: 25).");
    OpenSim_DECLARE_PROPERTY(parallel, int,
            "The number of threads used to parallelize the path fitting "
            "process (default: two fewer than the number of available "
            "hardware threads).");
    OpenSim_DECLARE_PROPERTY(
            latin_hypercube_algorithm, std::string,
            "The Latin hypercube sampling algorithm used to sample coordinate "
            "values for path fitting (default: \"random\").");

    void constructProperties();

    // PATH FITTING PIPELINE
    typedef std::unordered_map<std::string, std::vector<std::string>>
            MomentArmMap;

    TimeSeriesTable sampleCoordinateValues(const TimeSeriesTable& values);

    static void computePathLengthsAndMomentArms(const Model& model,
            const TimeSeriesTable& coordinateValues, int numThreads,
            TimeSeriesTable& pathLengths, TimeSeriesTable& momentArms);

    void filterSampledData(const Model& model,
            TimeSeriesTable& coordinateValues, TimeSeriesTable& pathLengths,
            TimeSeriesTable& momentArms, MomentArmMap& momentArmMap);

    Set<FunctionBasedPath> fitPolynomialCoefficients(const Model& model,
            const TimeSeriesTable& coordinateValues,
            const TimeSeriesTable& pathLengths,
            const TimeSeriesTable& momentArms,
            const MomentArmMap& momentArmMap);

    // HELPER FUNCTIONS
    /**
     * Get the (canonicalized) absolute directory containing the file from
     * which this tool was loaded. If the `FunctionBasedPathFitter` was not
     * loaded from a file, this returns an empty string.
     */
    std::string getDocumentDirectory() const;

    /**
     * Remove columns from the `momentArms` table that do not correspond to
     * entries in the `momentArmMap`.
     */
    static void removeMomentArmColumns(TimeSeriesTable& momentArms,
            const MomentArmMap& momentArmMap);

    /**
     * Get the RMS errors between two sets of path lengths and moment arms
     * computed from a model with FunctionBasedPaths and the original model. The
     * `modelFitted` argument must be the model with the FunctionBasedPaths.
     */
    static void computeFittingErrors(const Model& modelFitted,
            const TimeSeriesTable& pathLengths,
            const TimeSeriesTable& momentArms,
            const TimeSeriesTable& pathLengthsFitted,
            const TimeSeriesTable& momentArmsFitted);

    // MEMBER VARIABLES
    std::unordered_map<std::string, SimTK::Vec2> m_coordinateBoundsMap;
    std::unordered_map<std::string, SimTK::Vec2> m_coordinateRangeMap;
    bool m_useStochasticEvolutionaryLHS = false;
};

} // namespace OpenSim

#endif // OPENSIM_POLYNOMIALPATHFITTER_H

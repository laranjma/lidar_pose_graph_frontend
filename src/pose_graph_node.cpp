#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

using geometry_msgs::msg::PoseStamped;
using nav_msgs::msg::Odometry;
using nav_msgs::msg::Path;
using sensor_msgs::msg::LaserScan;

namespace {

constexpr double kPi = 3.14159265358979323846;

double quatToYaw(double qx, double qy, double qz, double qw) {
  const double siny_cosp = 2.0 * (qw * qz + qx * qy);
  const double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
  return std::atan2(siny_cosp, cosy_cosp);
}

double wrapAngle(double angle) {
  double wrapped = std::fmod(angle + kPi, 2.0 * kPi);
  if (wrapped < 0.0) {
    wrapped += 2.0 * kPi;
  }
  return wrapped - kPi;
}

PoseStamped pose2ToPoseStamped(const gtsam::Pose2 & pose, const std::string & frame_id, int32_t sec, uint32_t nsec) {
  PoseStamped ps;
  ps.header.frame_id = frame_id;
  ps.header.stamp.sec = sec;
  ps.header.stamp.nanosec = nsec;
  ps.pose.position.x = pose.x();
  ps.pose.position.y = pose.y();
  ps.pose.position.z = 0.0;

  const double yaw = pose.theta();
  ps.pose.orientation.z = std::sin(yaw / 2.0);
  ps.pose.orientation.w = std::cos(yaw / 2.0);
  return ps;
}

std::vector<Eigen::Vector2d> scanToXY(const LaserScan & scan) {
  std::vector<Eigen::Vector2d> points;
  points.reserve(scan.ranges.size());

  const double angle_min = static_cast<double>(scan.angle_min);
  const double angle_inc = static_cast<double>(scan.angle_increment);
  const double range_min = static_cast<double>(scan.range_min);
  const double range_max = static_cast<double>(scan.range_max);

  for (size_t i = 0; i < scan.ranges.size(); ++i) {
    const double r = static_cast<double>(scan.ranges[i]);
    if (!std::isfinite(r)) {
      continue;
    }
    if (r < range_min || r > range_max) {
      continue;
    }

    const double a = angle_min + static_cast<double>(i) * angle_inc;
    points.emplace_back(r * std::cos(a), r * std::sin(a));
  }
  return points;
}

std::vector<Eigen::Vector2d> downsample(const std::vector<Eigen::Vector2d> & points, size_t max_points) {
  if (points.size() <= max_points || max_points == 0) {
    return points;
  }

  const size_t step = static_cast<size_t>(std::ceil(static_cast<double>(points.size()) / static_cast<double>(max_points)));
  std::vector<Eigen::Vector2d> sampled;
  sampled.reserve(max_points);
  for (size_t i = 0; i < points.size(); i += step) {
    sampled.push_back(points[i]);
  }
  return sampled;
}

bool bestFitTransform2D(
  const std::vector<Eigen::Vector2d> & a,
  const std::vector<Eigen::Vector2d> & b,
  Eigen::Matrix2d & r,
  Eigen::Vector2d & t)
{
  if (a.empty() || b.empty() || a.size() != b.size()) {
    return false;
  }

  Eigen::Vector2d centroid_a = Eigen::Vector2d::Zero();
  Eigen::Vector2d centroid_b = Eigen::Vector2d::Zero();
  for (size_t i = 0; i < a.size(); ++i) {
    centroid_a += a[i];
    centroid_b += b[i];
  }
  centroid_a /= static_cast<double>(a.size());
  centroid_b /= static_cast<double>(b.size());

  Eigen::Matrix2d h = Eigen::Matrix2d::Zero();
  for (size_t i = 0; i < a.size(); ++i) {
    const Eigen::Vector2d aa = a[i] - centroid_a;
    const Eigen::Vector2d bb = b[i] - centroid_b;
    h += bb * aa.transpose();
  }

  Eigen::JacobiSVD<Eigen::Matrix2d> svd(h, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix2d u = svd.matrixU();
  Eigen::Matrix2d v = svd.matrixV();
  r = v * u.transpose();

  if (r.determinant() < 0.0) {
    v.col(1) *= -1.0;
    r = v * u.transpose();
  }
  t = centroid_a - (r * centroid_b);
  return true;
}

double yawFromR(const Eigen::Matrix2d & r) {
  return std::atan2(r(1, 0), r(0, 0));
}

std::optional<gtsam::Pose2> computePoseLS(const LaserScan::ConstSharedPtr & scan_i, const LaserScan::ConstSharedPtr & scan_j) {
  if (!scan_i || !scan_j) {
    return std::nullopt;
  }
  if (scan_i->ranges.empty() || scan_j->ranges.empty()) {
    return std::nullopt;
  }
  if (static_cast<double>(scan_i->angle_increment) == 0.0 || static_cast<double>(scan_j->angle_increment) == 0.0) {
    return std::nullopt;
  }

  auto ref = scanToXY(*scan_i);
  auto sens = scanToXY(*scan_j);
  if (ref.size() < 30 || sens.size() < 30) {
    return std::nullopt;
  }

  constexpr size_t kMaxPoints = 700;
  ref = downsample(ref, kMaxPoints);
  sens = downsample(sens, kMaxPoints);

  constexpr int kMaxIters = 50;
  constexpr double kMaxCorrDist = 2.0;
  constexpr int kMinCorr = 40;
  constexpr double kEpsTrans = 1e-4;
  constexpr double kEpsRot = 1e-4;
  const double max_corr_d2 = kMaxCorrDist * kMaxCorrDist;

  Eigen::Matrix2d r_total = Eigen::Matrix2d::Identity();
  Eigen::Vector2d t_total = Eigen::Vector2d::Zero();
  std::optional<double> prev_err;

  std::vector<Eigen::Vector2d> sens_tf;
  sens_tf.resize(sens.size());

  for (int iter = 0; iter < kMaxIters; ++iter) {
    for (size_t i = 0; i < sens.size(); ++i) {
      sens_tf[i] = (r_total * sens[i]) + t_total;
    }

    std::vector<Eigen::Vector2d> a;
    std::vector<Eigen::Vector2d> b_current;
    a.reserve(sens_tf.size());
    b_current.reserve(sens_tf.size());

    int good_count = 0;
    double d2_sum = 0.0;

    for (const auto & q : sens_tf) {
      double best_d2 = std::numeric_limits<double>::max();
      int best_idx = -1;
      for (size_t k = 0; k < ref.size(); ++k) {
        const double d2 = (q - ref[k]).squaredNorm();
        if (d2 < best_d2) {
          best_d2 = d2;
          best_idx = static_cast<int>(k);
        }
      }

      if (best_idx >= 0 && best_d2 <= max_corr_d2) {
        ++good_count;
        d2_sum += best_d2;
        a.push_back(ref[static_cast<size_t>(best_idx)]);
        b_current.push_back(q);
      }
    }

    if (good_count < kMinCorr) {
      return std::nullopt;
    }

    Eigen::Matrix2d r_inc = Eigen::Matrix2d::Identity();
    Eigen::Vector2d t_inc = Eigen::Vector2d::Zero();
    if (!bestFitTransform2D(a, b_current, r_inc, t_inc)) {
      return std::nullopt;
    }

    r_total = r_inc * r_total;
    t_total = (r_inc * t_total) + t_inc;

    const double mean_err = std::sqrt(d2_sum / static_cast<double>(good_count));
    if (prev_err && std::abs(*prev_err - mean_err) < 1e-5) {
      break;
    }
    prev_err = mean_err;

    if (std::abs(t_inc.x()) < kEpsTrans && std::abs(t_inc.y()) < kEpsTrans && std::abs(yawFromR(r_inc)) < kEpsRot) {
      break;
    }
  }

  if (!prev_err || *prev_err > 0.5) {
    return std::nullopt;
  }

  const double dx = t_total.x();
  const double dy = t_total.y();
  const double dth = wrapAngle(yawFromR(r_total));

  if (std::hypot(dx, dy) > 3.0 || std::abs(dth) > (120.0 * kPi / 180.0)) {
    return std::nullopt;
  }

  return gtsam::Pose2(dx, dy, dth);
}

struct Stamp {
  int32_t sec{0};
  uint32_t nsec{0};

  [[nodiscard]] int64_t ns() const {
    return static_cast<int64_t>(sec) * 1000000000LL + static_cast<int64_t>(nsec);
  }
};

struct Keyframe {
  int idx{0};
  Stamp stamp{};
  gtsam::Pose2 odom_pose{};
  LaserScan::ConstSharedPtr scan{nullptr};
};

struct LoopClosureConstraint {
  int i{0};
  int j{0};
  gtsam::Pose2 z_ij{};
  gtsam::SharedNoiseModel noise;
};

class ILoopClosure {
public:
  virtual ~ILoopClosure() = default;

  virtual std::vector<LoopClosureConstraint> onNewKeyframe(
    const Keyframe & new_kf,
    const std::vector<Keyframe> & keyframes,
    const gtsam::Values * current_estimate) = 0;
};

class LoopClosureModule final : public ILoopClosure {
public:
  explicit LoopClosureModule(
    gtsam::SharedNoiseModel noise,
    int min_idx_separation = 20,
    double search_radius_m = 1.5,
    int max_candidates = 3,
    double max_trans_error_m = 1.0,
    double max_rot_error_rad = 35.0 * kPi / 180.0)
  : noise_(std::move(noise)),
    min_idx_separation_(min_idx_separation),
    search_radius_m_(search_radius_m),
    max_candidates_(max_candidates),
    max_trans_error_m_(max_trans_error_m),
    max_rot_error_rad_(max_rot_error_rad) {}

  std::vector<LoopClosureConstraint> onNewKeyframe(
    const Keyframe & new_kf,
    const std::vector<Keyframe> & keyframes,
    const gtsam::Values * current_estimate) override
  {
    if (!new_kf.scan || keyframes.size() <= 1) {
      return {};
    }

    const int j_idx = new_kf.idx;
    const gtsam::Pose2 j_pose = poseForKeyframe(new_kf, current_estimate);

    struct Candidate {
      double dist{0.0};
      const Keyframe * keyframe{nullptr};
      gtsam::Pose2 pose{};
    };
    std::vector<Candidate> candidates;
    candidates.reserve(keyframes.size());

    for (size_t k = 0; k + 1 < keyframes.size(); ++k) {
      const auto & kf = keyframes[k];
      if ((j_idx - kf.idx) < min_idx_separation_) {
        continue;
      }
      if (!kf.scan) {
        continue;
      }

      const gtsam::Pose2 i_pose = poseForKeyframe(kf, current_estimate);
      const double dist = std::hypot(i_pose.x() - j_pose.x(), i_pose.y() - j_pose.y());
      if (dist > search_radius_m_) {
        continue;
      }
      candidates.push_back(Candidate{dist, &kf, i_pose});
    }

    if (candidates.empty()) {
      return {};
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate & lhs, const Candidate & rhs) {
      return lhs.dist < rhs.dist;
    });

    const size_t n = std::min(static_cast<size_t>(std::max(max_candidates_, 0)), candidates.size());
    for (size_t k = 0; k < n; ++k) {
      const auto & candidate = candidates[k];
      auto z_ij = computePoseLS(candidate.keyframe->scan, new_kf.scan);
      if (!z_ij) {
        continue;
      }

      const gtsam::Pose2 pred_ij = candidate.pose.between(j_pose);
      const gtsam::Pose2 err = pred_ij.between(*z_ij);
      if (std::hypot(err.x(), err.y()) > max_trans_error_m_) {
        continue;
      }
      if (std::abs(wrapAngle(err.theta())) > max_rot_error_rad_) {
        continue;
      }

      return {LoopClosureConstraint{candidate.keyframe->idx, j_idx, *z_ij, noise_}};
    }

    return {};
  }

private:
  static gtsam::Pose2 poseForKeyframe(const Keyframe & kf, const gtsam::Values * current_estimate) {
    if (current_estimate != nullptr) {
      const gtsam::Key key = gtsam::Symbol('x', static_cast<size_t>(kf.idx));
      if (current_estimate->exists(key)) {
        return current_estimate->at<gtsam::Pose2>(key);
      }
    }
    return kf.odom_pose;
  }

  gtsam::SharedNoiseModel noise_;
  int min_idx_separation_{20};
  double search_radius_m_{1.5};
  int max_candidates_{3};
  double max_trans_error_m_{1.0};
  double max_rot_error_rad_{35.0 * kPi / 180.0};
};

class NoLoopClosure final : public ILoopClosure {
public:
  std::vector<LoopClosureConstraint> onNewKeyframe(
    const Keyframe &,
    const std::vector<Keyframe> &,
    const gtsam::Values *) override
  {
    return {};
  }
};

class BackendOptimizer {
public:
  BackendOptimizer() {
    gtsam::ISAM2Params params;
    params.relinearizeThreshold = 0.01;
    params.relinearizeSkip = 1;
    isam_ = std::make_unique<gtsam::ISAM2>(params);
  }

  void addPrior(int idx, const gtsam::Pose2 & pose, const gtsam::SharedNoiseModel & noise) {
    const gtsam::Key key = gtsam::Symbol('x', static_cast<size_t>(idx));
    graph_.add(gtsam::PriorFactor<gtsam::Pose2>(key, pose, noise));
    if (!init_.exists(key)) {
      init_.insert(key, pose);
    }
  }

  void addBetween(
    int i,
    int j,
    const gtsam::Pose2 & z_ij,
    const gtsam::SharedNoiseModel & noise,
    const gtsam::Pose2 & initial_guess_for_j)
  {
    const gtsam::Key ki = gtsam::Symbol('x', static_cast<size_t>(i));
    const gtsam::Key kj = gtsam::Symbol('x', static_cast<size_t>(j));
    graph_.add(gtsam::BetweenFactor<gtsam::Pose2>(ki, kj, z_ij, noise));
    if (!init_.exists(kj)) {
      init_.insert(kj, initial_guess_for_j);
    }
  }

  void update() {
    isam_->update(graph_, init_);
    graph_.resize(0);
    init_.clear();
  }

  gtsam::Values getEstimate() const {
    return isam_->calculateEstimate();
  }

private:
  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values init_;
  std::unique_ptr<gtsam::ISAM2> isam_;
};

struct BackendAction {
  enum class Kind {
    Prior,
    Between
  };

  Kind kind{Kind::Prior};
  int idx{0};
  int i{0};
  int j{0};
  gtsam::Pose2 pose{0.0, 0.0, 0.0};
  gtsam::Pose2 z_ij{0.0, 0.0, 0.0};
  gtsam::SharedNoiseModel noise;
  gtsam::Pose2 init_j{0.0, 0.0, 0.0};
};

class Frontend {
public:
  Frontend(
    double trans_thresh_m,
    double rot_thresh_rad,
    gtsam::SharedNoiseModel noise_prior,
    gtsam::SharedNoiseModel noise_odom,
    std::unique_ptr<ILoopClosure> loop_closure)
  : trans_thresh_(trans_thresh_m),
    rot_thresh_(rot_thresh_rad),
    noise_prior_(std::move(noise_prior)),
    noise_odom_(std::move(noise_odom)),
    loop_closure_(std::move(loop_closure)) {}

  void onScan(const LaserScan::ConstSharedPtr & scan) {
    last_scan_ = scan;
    const int64_t t_ns = static_cast<int64_t>(scan->header.stamp.sec) * 1000000000LL +
      static_cast<int64_t>(scan->header.stamp.nanosec);
    pending_scans_by_time_[t_ns] = scan;

    if (pending_scans_by_time_.size() > 5000) {
      for (int i = 0; i < 1000 && !pending_scans_by_time_.empty(); ++i) {
        pending_scans_by_time_.erase(pending_scans_by_time_.begin());
      }
    }
  }

  std::pair<std::optional<Keyframe>, std::vector<BackendAction>> maybeCreateKeyframe(
    const gtsam::Pose2 & odom_pose,
    const Stamp & stamp,
    const gtsam::Values * backend_estimate)
  {
    std::vector<BackendAction> actions;

    if (keyframes_.empty()) {
      Keyframe kf0 = makeKeyframe(0, odom_pose, stamp);
      keyframes_.push_back(kf0);

      BackendAction action;
      action.kind = BackendAction::Kind::Prior;
      action.idx = 0;
      action.pose = odom_pose;
      action.noise = noise_prior_;
      actions.push_back(action);
      return {kf0, actions};
    }

    const Keyframe & last_kf = keyframes_.back();
    const gtsam::Pose2 rel = last_kf.odom_pose.between(odom_pose);
    const double trans = std::hypot(rel.x(), rel.y());
    const double rot = std::abs(wrapAngle(rel.theta()));

    if (trans < trans_thresh_ && rot < rot_thresh_) {
      return {std::nullopt, actions};
    }

    const int new_idx = last_kf.idx + 1;
    Keyframe new_kf = makeKeyframe(new_idx, odom_pose, stamp);
    keyframes_.push_back(new_kf);

    const gtsam::Pose2 init_j = composeFromBackendOrFallback(
      backend_estimate, last_kf.idx, last_kf.odom_pose, rel);

    auto z_ij = computePoseLS(last_kf.scan, new_kf.scan);
    if (!z_ij) {
      z_ij = rel;
    }

    BackendAction between_action;
    between_action.kind = BackendAction::Kind::Between;
    between_action.i = last_kf.idx;
    between_action.j = new_idx;
    between_action.z_ij = *z_ij;
    between_action.noise = noise_odom_;
    between_action.init_j = init_j;
    actions.push_back(between_action);

    for (const auto & lc : loop_closure_->onNewKeyframe(new_kf, keyframes_, backend_estimate)) {
      BackendAction loop_action;
      loop_action.kind = BackendAction::Kind::Between;
      loop_action.i = lc.i;
      loop_action.j = lc.j;
      loop_action.z_ij = lc.z_ij;
      loop_action.noise = lc.noise;
      loop_action.init_j = init_j;
      actions.push_back(loop_action);
    }

    return {new_kf, actions};
  }

  [[nodiscard]] const std::vector<Keyframe> & keyframes() const {
    return keyframes_;
  }

private:
  Keyframe makeKeyframe(int idx, const gtsam::Pose2 & odom_pose, const Stamp & stamp) const {
    LaserScan::ConstSharedPtr scan = last_scan_;
    const auto it = pending_scans_by_time_.find(stamp.ns());
    if (it != pending_scans_by_time_.end()) {
      scan = it->second;
    }
    return Keyframe{idx, stamp, odom_pose, scan};
  }

  static gtsam::Pose2 composeFromBackendOrFallback(
    const gtsam::Values * backend_estimate,
    int prev_idx,
    const gtsam::Pose2 & prev_fallback,
    const gtsam::Pose2 & rel)
  {
    if (backend_estimate != nullptr) {
      const gtsam::Key key_prev = gtsam::Symbol('x', static_cast<size_t>(prev_idx));
      if (backend_estimate->exists(key_prev)) {
        const gtsam::Pose2 prev_opt = backend_estimate->at<gtsam::Pose2>(key_prev);
        return prev_opt.compose(rel);
      }
    }
    return prev_fallback.compose(rel);
  }

  double trans_thresh_{0.2};
  double rot_thresh_{0.2};
  gtsam::SharedNoiseModel noise_prior_;
  gtsam::SharedNoiseModel noise_odom_;
  std::unique_ptr<ILoopClosure> loop_closure_;

  std::vector<Keyframe> keyframes_;
  std::map<int64_t, LaserScan::ConstSharedPtr> pending_scans_by_time_;
  LaserScan::ConstSharedPtr last_scan_{nullptr};
};

class PoseGraphNode : public rclcpp::Node {
public:
  PoseGraphNode()
  : Node("pose_graph_node") {
    declare_parameter<std::string>("odom_topic", "/odom");
    declare_parameter<std::string>("scan_topic", "/scan");
    const std::string odom_topic = get_parameter("odom_topic").as_string();
    const std::string scan_topic = get_parameter("scan_topic").as_string();

    declare_parameter<double>("trans_thresh_m", 0.2);
    declare_parameter<double>("rot_thresh_rad", 0.2);

    declare_parameter<std::vector<double>>("prior_sigmas", {1e-3, 1e-3, 1e-3});
    declare_parameter<std::vector<double>>("odom_sigmas", {0.1, 0.1, 0.20});
    declare_parameter<std::vector<double>>("loop_sigmas", {0.2, 0.2, 0.30});

    declare_parameter<bool>("loop_enable", true);
    declare_parameter<int>("loop_min_idx_separation", 5);
    declare_parameter<double>("loop_search_radius_m", 1.5);
    declare_parameter<int>("loop_max_candidates", 3);
    declare_parameter<double>("loop_max_trans_error_m", 3.0);
    declare_parameter<double>("loop_max_rot_error_rad", 1.57);

    const double trans_thresh = get_parameter("trans_thresh_m").as_double();
    const double rot_thresh = get_parameter("rot_thresh_rad").as_double();

    const auto prior_sigmas = validateSigmas("prior_sigmas", get_parameter("prior_sigmas").as_double_array(), {1e-3, 1e-3, 1e-3});
    const auto odom_sigmas = validateSigmas("odom_sigmas", get_parameter("odom_sigmas").as_double_array(), {0.1, 0.1, 0.20});
    const auto loop_sigmas = validateSigmas("loop_sigmas", get_parameter("loop_sigmas").as_double_array(), {0.2, 0.2, 0.30});

    const bool loop_enable = get_parameter("loop_enable").as_bool();
    const int loop_min_idx_separation = get_parameter("loop_min_idx_separation").as_int();
    const double loop_search_radius_m = get_parameter("loop_search_radius_m").as_double();
    const int loop_max_candidates = get_parameter("loop_max_candidates").as_int();
    const double loop_max_trans_error_m = get_parameter("loop_max_trans_error_m").as_double();
    const double loop_max_rot_error_rad = get_parameter("loop_max_rot_error_rad").as_double();

    const auto noise_prior = gtsam::noiseModel::Diagonal::Sigmas(makeVector3(prior_sigmas));
    const auto noise_odom = gtsam::noiseModel::Diagonal::Sigmas(makeVector3(odom_sigmas));
    const auto noise_loop = gtsam::noiseModel::Diagonal::Sigmas(makeVector3(loop_sigmas));

    std::unique_ptr<ILoopClosure> loop_closure;
    if (loop_enable) {
      loop_closure = std::make_unique<LoopClosureModule>(
        noise_loop,
        loop_min_idx_separation,
        loop_search_radius_m,
        loop_max_candidates,
        loop_max_trans_error_m,
        loop_max_rot_error_rad);
    } else {
      loop_closure = std::make_unique<NoLoopClosure>();
    }

    frontend_ = std::make_unique<Frontend>(
      trans_thresh, rot_thresh, noise_prior, noise_odom, std::move(loop_closure));

    path_raw_pub_ = create_publisher<Path>("/pose_graph/path_raw", 10);
    path_opt_pub_ = create_publisher<Path>("/pose_graph/path_opt", 10);

    scan_sub_ = create_subscription<LaserScan>(
      scan_topic,
      50,
      [this](const LaserScan::ConstSharedPtr msg) { onScan(msg); });
    odom_sub_ = create_subscription<Odometry>(
      odom_topic,
      50,
      [this](const Odometry::ConstSharedPtr msg) { onOdom(msg); });

    RCLCPP_INFO(
      get_logger(),
      "Subscribed to odom=%s, scan=%s; keyframe thresholds: trans=%.3f m, rot=%.3f rad",
      odom_topic.c_str(), scan_topic.c_str(), trans_thresh, rot_thresh);
  }

private:
  static gtsam::Vector3 makeVector3(const std::array<double, 3> & values) {
    gtsam::Vector3 vec;
    vec << values[0], values[1], values[2];
    return vec;
  }

  std::array<double, 3> validateSigmas(
    const std::string & name,
    const std::vector<double> & sigmas,
    const std::array<double, 3> & fallback)
  {
    if (sigmas.size() != 3) {
      RCLCPP_WARN(
        get_logger(),
        "Parameter %s must have 3 values, got %zu. Using defaults.",
        name.c_str(),
        sigmas.size());
      return fallback;
    }
    return {sigmas[0], sigmas[1], sigmas[2]};
  }

  void onScan(const LaserScan::ConstSharedPtr & msg) {
    frontend_->onScan(msg);
  }

  void onOdom(const Odometry::ConstSharedPtr & msg) {
    const auto & p = msg->pose.pose.position;
    const auto & q = msg->pose.pose.orientation;
    const double yaw = quatToYaw(q.x, q.y, q.z, q.w);
    const gtsam::Pose2 odom_pose(p.x, p.y, yaw);
    const Stamp stamp{msg->header.stamp.sec, msg->header.stamp.nanosec};

    std::optional<gtsam::Values> backend_estimate;
    try {
      backend_estimate = backend_.getEstimate();
    } catch (const std::exception &) {
      backend_estimate = std::nullopt;
    }

    const gtsam::Values * est_ptr = backend_estimate ? &(*backend_estimate) : nullptr;
    auto [new_kf, actions] = frontend_->maybeCreateKeyframe(odom_pose, stamp, est_ptr);
    if (!new_kf) {
      return;
    }

    for (const auto & action : actions) {
      if (action.kind == BackendAction::Kind::Prior) {
        backend_.addPrior(action.idx, action.pose, action.noise);
      } else if (action.kind == BackendAction::Kind::Between) {
        backend_.addBetween(action.i, action.j, action.z_ij, action.noise, action.init_j);
      }
    }

    backend_.update();
    publishPaths();
  }

  void publishPaths() {
    const auto & keyframes = frontend_->keyframes();
    if (keyframes.empty()) {
      return;
    }

    constexpr const char * kFrameId = "odom";
    const Stamp & last_stamp = keyframes.back().stamp;

    Path path_raw;
    path_raw.header.frame_id = kFrameId;
    path_raw.header.stamp.sec = last_stamp.sec;
    path_raw.header.stamp.nanosec = last_stamp.nsec;

    for (const auto & kf : keyframes) {
      path_raw.poses.push_back(pose2ToPoseStamped(kf.odom_pose, kFrameId, kf.stamp.sec, kf.stamp.nsec));
    }
    path_raw_pub_->publish(path_raw);

    Path path_opt;
    path_opt.header = path_raw.header;

    gtsam::Values estimate;
    try {
      estimate = backend_.getEstimate();
    } catch (const std::exception &) {
      path_opt_pub_->publish(path_opt);
      return;
    }

    for (const auto & kf : keyframes) {
      const gtsam::Key key = gtsam::Symbol('x', static_cast<size_t>(kf.idx));
      if (!estimate.exists(key)) {
        continue;
      }

      const auto pose = estimate.at<gtsam::Pose2>(key);
      path_opt.poses.push_back(pose2ToPoseStamped(pose, kFrameId, kf.stamp.sec, kf.stamp.nsec));
    }

    path_opt_pub_->publish(path_opt);
  }

  std::unique_ptr<Frontend> frontend_;
  BackendOptimizer backend_;

  rclcpp::Publisher<Path>::SharedPtr path_raw_pub_;
  rclcpp::Publisher<Path>::SharedPtr path_opt_pub_;
  rclcpp::Subscription<LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<Odometry>::SharedPtr odom_sub_;
};

}  // namespace

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PoseGraphNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

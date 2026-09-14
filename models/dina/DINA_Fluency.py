# coding: utf-8
# DINA + Fluency (post-logit calibration style)

import logging
import numpy as np
from tqdm import tqdm
import pickle
from EduCDM import CDM


EPS = 1e-9
PROB_MIN = 1e-3
PROB_MAX = 1 - 1e-3


def initial_all_knowledge_state(know_num):
    state_num = 2 ** know_num
    all_states = np.zeros((state_num, know_num), dtype=np.float64)
    for i in range(state_num):
        k, quotient, residue = 1, i // 2, i % 2
        while True:
            all_states[i, know_num - k] = residue
            if quotient <= 0:
                break
            quotient, residue = quotient // 2, quotient % 2
            k += 1
    return all_states


def init_parameters(stu_num, prob_num):
    slip = np.zeros(shape=prob_num, dtype=np.float64) + 0.2
    guess = np.zeros(shape=prob_num, dtype=np.float64) + 0.2
    theta = np.zeros(shape=stu_num, dtype=np.int64)
    return theta, slip, guess


def safe_clip_prob(x, low=PROB_MIN, high=PROB_MAX):
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.5, posinf=high, neginf=low)
    return np.clip(x, low, high)


def safe_logit(p):
    p = safe_clip_prob(p)
    return np.log(p + EPS) - np.log(1 - p + EPS)


def safe_sigmoid(x):
    x = np.nan_to_num(x, nan=0.0, posinf=30.0, neginf=-30.0)
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


class DINAFluency(CDM):
    """
    DINA + Fluency

    Fluency is modeled as a student-level proxy score:
        flu_i = w1 * log1p(opportunity_i) + w2 * acc_prior_i + w3 * stability_i + b

    Final probability:
        P = sigmoid(logit(P_DINA) + gamma * flu_i)

    Notes:
    - EM updates theta/slip/guess exactly as in vanilla DINA
    - fluency is a post-logit calibration term
    - this is much more stable than injecting fluency directly into EM updates
    """

    def __init__(self, R, q_m, stu_num, prob_num, know_num,
                 skip_value=-1,
                 opportunity=None,
                 acc_prior=None,
                 stability=None,
                 gamma=1.0,
                 flu_weights=None,
                 flu_bias=0.0):
        self.R = np.asarray(R, dtype=np.float64)
        self.q_m = np.asarray(q_m, dtype=np.float64)
        self.state_num = 2 ** know_num
        self.skip_value = skip_value
        self.stu_num, self.prob_num, self.know_num = stu_num, prob_num, know_num

        self.theta, self.slip, self.guess = init_parameters(stu_num, prob_num)
        self.all_states = initial_all_knowledge_state(know_num)

        state_prob = np.transpose(
            np.sum(self.q_m, axis=1, keepdims=True) - np.dot(self.q_m, np.transpose(self.all_states))
        )
        self.eta = 1 - (state_prob > 0)
        self.eta = self.eta.astype(np.float64)

        # ---------- fluency proxies ----------
        # student-level vectors, shape = (stu_num,)
        self.opportunity = self._prepare_student_vector(opportunity, stu_num, default=0.0)
        self.acc_prior = self._prepare_student_vector(acc_prior, stu_num, default=0.5)
        self.stability = self._prepare_student_vector(stability, stu_num, default=0.5)

        self.gamma = float(gamma)
        if flu_weights is None:
            # [opp, acc, sta]
            flu_weights = np.array([0.2, 1.0, 0.5], dtype=np.float64)
        self.flu_weights = np.asarray(flu_weights, dtype=np.float64).reshape(3,)
        self.flu_bias = float(flu_bias)

    @staticmethod
    def _prepare_student_vector(x, stu_num, default=0.0):
        if x is None:
            return np.zeros(stu_num, dtype=np.float64) + default
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if len(x) != stu_num:
            raise ValueError(f"Expected student-level vector length={stu_num}, got {len(x)}")
        x = np.nan_to_num(x, nan=default, posinf=default, neginf=default)
        return x

    def _compute_student_fluency(self):
        """
        Compute student-level fluency score.
        Higher is better.
        """
        opp = np.log1p(np.clip(self.opportunity, 0.0, None))
        acc = np.clip(self.acc_prior, 0.0, 1.0)
        sta = np.clip(self.stability, 0.0, 1.0)

        flu = (
            self.flu_weights[0] * opp +
            self.flu_weights[1] * acc +
            self.flu_weights[2] * sta +
            self.flu_bias
        )
        flu = np.nan_to_num(flu, nan=0.0, posinf=10.0, neginf=-10.0)
        return np.clip(flu, -10.0, 10.0)

    def train(self, epoch, epsilon) -> ...:
        like = np.zeros((self.stu_num, self.state_num), dtype=np.float64)
        post = np.zeros((self.stu_num, self.state_num), dtype=np.float64)

        theta = np.copy(self.theta)
        slip = safe_clip_prob(np.copy(self.slip))
        guess = safe_clip_prob(np.copy(self.guess))

        obs_mask = (self.R != self.skip_value).astype(np.float64)
        tmp_R = np.copy(self.R)
        tmp_R[self.R == self.skip_value] = 0.0

        a_g, b_g = 4.0, 4.0
        a_s, b_s = 4.0, 4.0

        for iteration in range(epoch):
            post_tmp = np.copy(post)
            slip_tmp = np.copy(slip)
            guess_tmp = np.copy(guess)

            answer_right = (1 - slip) * self.eta + guess * (1 - self.eta)
            answer_right = safe_clip_prob(answer_right)

            # E-step
            for s in range(self.state_num):
                log_like = (
                    np.log(answer_right[s, :] + EPS) * tmp_R
                    + np.log(1 - answer_right[s, :] + EPS) * (1 - tmp_R)
                )
                log_like[self.R == self.skip_value] = 0.0

                row_log_like = np.sum(log_like, axis=1)
                row_log_like = np.nan_to_num(row_log_like, nan=-1e12, posinf=1e12, neginf=-1e12)
                like[:, s] = np.exp(np.clip(row_log_like, -700, 700))

            like = np.nan_to_num(like, nan=EPS, posinf=EPS, neginf=EPS)
            like = np.maximum(like, EPS)

            den = np.sum(like, axis=1, keepdims=True)
            den = np.nan_to_num(den, nan=EPS, posinf=EPS, neginf=EPS)
            den = np.maximum(den, EPS)

            post = like / den
            post = np.nan_to_num(
                post,
                nan=1.0 / self.state_num,
                posinf=1.0 / self.state_num,
                neginf=1.0 / self.state_num
            )
            post = np.maximum(post, EPS)
            post = post / np.maximum(np.sum(post, axis=1, keepdims=True), EPS)

            # M-step
            r_jl = np.dot(post.T, tmp_R)
            n_jl = np.dot(post.T, obs_mask)

            r_jl_0 = np.sum(r_jl * (1 - self.eta), axis=0)
            r_jl_1 = np.sum(r_jl * self.eta, axis=0)

            n_jl_0 = np.sum(n_jl * (1 - self.eta), axis=0)
            n_jl_1 = np.sum(n_jl * self.eta, axis=0)

            n_jl_0 = np.maximum(np.nan_to_num(n_jl_0, nan=EPS, posinf=EPS, neginf=EPS), EPS)
            n_jl_1 = np.maximum(np.nan_to_num(n_jl_1, nan=EPS, posinf=EPS, neginf=EPS), EPS)

            guess = (r_jl_0 + a_g - 1.0) / (n_jl_0 + a_g + b_g - 2.0)
            slip = (n_jl_1 - r_jl_1 + a_s - 1.0) / (n_jl_1 + a_s + b_s - 2.0)

            guess = safe_clip_prob(guess)
            slip = safe_clip_prob(slip)

            change = max(
                np.max(np.abs(post - post_tmp)),
                np.max(np.abs(slip - slip_tmp)),
                np.max(np.abs(guess - guess_tmp))
            )

            theta = np.argmax(post, axis=1)

            logging.info(
                "iteration %d, change=%.10f, slip[min=%.6f,max=%.6f,mean=%.6f], guess[min=%.6f,max=%.6f,mean=%.6f]",
                iteration + 1,
                change,
                float(np.min(slip)), float(np.max(slip)), float(np.mean(slip)),
                float(np.min(guess)), float(np.max(guess)), float(np.mean(guess))
            )

            if iteration > 20 and change < epsilon:
                break

        self.theta = theta
        self.slip = safe_clip_prob(slip)
        self.guess = safe_clip_prob(guess)

    def _build_pred_score_matrix(self):
        """
        Build student-item probability matrix with fluency correction.
        Shape: (stu_num, prob_num)
        """
        slip = safe_clip_prob(self.slip)
        guess = safe_clip_prob(self.guess)

        # state-item score matrix: (state_num, prob_num)
        base_state_score = (1 - slip) * self.eta + guess * (1 - self.eta)
        base_state_score = safe_clip_prob(base_state_score)

        # student state selection -> student-item score
        base_student_score = base_state_score[self.theta, :]   # (stu_num, prob_num)

        # student-level fluency
        flu = self._compute_student_fluency().reshape(-1, 1)   # (stu_num, 1)

        # logit fusion
        score_logit = safe_logit(base_student_score)
        score_logit = score_logit + self.gamma * flu

        pred_score = safe_sigmoid(score_logit)
        pred_score = safe_clip_prob(pred_score)
        return pred_score

    def eval(self, test_data) -> tuple:
        pred_score = self._build_pred_score_matrix()

        test_rmse, test_mae = [], []
        for i in tqdm(test_data, "evaluating"):
            stu, test_id, true_score = i["user_id"], i["item_id"], i["score"]
            pred = pred_score[stu, test_id]
            pred = float(np.nan_to_num(pred, nan=0.5, posinf=0.999, neginf=0.001))
            test_rmse.append((pred - true_score) ** 2)
            test_mae.append(abs(pred - true_score))
        return np.sqrt(np.average(test_rmse)), np.average(test_mae)

    def predict_records(self, records, stu_id=None):
        """
        Predict probability for a student's response record vector.

        records: shape (prob_num,)
        stu_id: optional, if provided, use this student's fluency
        """
        records = np.asarray(records, dtype=np.float64)

        slip = safe_clip_prob(self.slip)
        guess = safe_clip_prob(self.guess)
        answer_right = (1 - slip) * self.eta + guess * (1 - self.eta)
        answer_right = safe_clip_prob(answer_right)

        log_like = records * np.log(answer_right + EPS) + (1 - records) * np.log(1 - answer_right + EPS)
        log_like[:, np.where(records == self.skip_value)[0]] = 0

        summed = np.sum(log_like, axis=1)
        summed = np.nan_to_num(summed, nan=-1e12, posinf=1e12, neginf=-1e12)

        max_like_id = np.argmax(np.exp(np.clip(summed, -700, 700)))

        base_score = answer_right[max_like_id, :]   # (prob_num,)

        if stu_id is not None:
            flu = self._compute_student_fluency()[stu_id]
            base_score = safe_sigmoid(safe_logit(base_score) + self.gamma * flu)
            base_score = safe_clip_prob(base_score)

        return base_score

    def transform(self, records):
        """
        Keep original transform behavior: return most likely knowledge state.
        """
        records = np.asarray(records, dtype=np.float64)

        slip = safe_clip_prob(self.slip)
        guess = safe_clip_prob(self.guess)
        answer_right = (1 - slip) * self.eta + guess * (1 - self.eta)
        answer_right = safe_clip_prob(answer_right)

        log_like = records * np.log(answer_right + EPS) + (1 - records) * np.log(1 - answer_right + EPS)
        log_like[:, np.where(records == self.skip_value)[0]] = 0

        summed = np.sum(log_like, axis=1)
        summed = np.nan_to_num(summed, nan=-1e12, posinf=1e12, neginf=-1e12)

        max_like_id = np.argmax(np.exp(np.clip(summed, -700, 700)))
        dia_state = self.all_states[max_like_id]
        return max_like_id, dia_state

    def update_fluency(self, opportunity=None, acc_prior=None, stability=None,
                       gamma=None, flu_weights=None, flu_bias=None):
        """
        Update fluency proxies / parameters after model construction.
        """
        if opportunity is not None:
            self.opportunity = self._prepare_student_vector(opportunity, self.stu_num, default=0.0)
        if acc_prior is not None:
            self.acc_prior = self._prepare_student_vector(acc_prior, self.stu_num, default=0.5)
        if stability is not None:
            self.stability = self._prepare_student_vector(stability, self.stu_num, default=0.5)
        if gamma is not None:
            self.gamma = float(gamma)
        if flu_weights is not None:
            self.flu_weights = np.asarray(flu_weights, dtype=np.float64).reshape(3,)
        if flu_bias is not None:
            self.flu_bias = float(flu_bias)

    def save(self, filepath):
        with open(filepath, "wb") as file:
            pickle.dump(
                {
                    "theta": self.theta,
                    "slip": self.slip,
                    "guess": self.guess,
                    "opportunity": self.opportunity,
                    "acc_prior": self.acc_prior,
                    "stability": self.stability,
                    "gamma": self.gamma,
                    "flu_weights": self.flu_weights,
                    "flu_bias": self.flu_bias,
                },
                file
            )
            logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        with open(filepath, "rb") as file:
            params = pickle.load(file)
            self.theta = params["theta"]
            self.slip = safe_clip_prob(params["slip"])
            self.guess = safe_clip_prob(params["guess"])

            self.opportunity = self._prepare_student_vector(
                params.get("opportunity", None), self.stu_num, default=0.0
            )
            self.acc_prior = self._prepare_student_vector(
                params.get("acc_prior", None), self.stu_num, default=0.5
            )
            self.stability = self._prepare_student_vector(
                params.get("stability", None), self.stu_num, default=0.5
            )
            self.gamma = float(params.get("gamma", 1.0))
            self.flu_weights = np.asarray(
                params.get("flu_weights", np.array([0.2, 1.0, 0.5], dtype=np.float64)),
                dtype=np.float64
            ).reshape(3,)
            self.flu_bias = float(params.get("flu_bias", 0.0))

            logging.info("load parameters from %s" % filepath)

    def inc_train(self, inc_train_data, epoch, epsilon):
        for i in inc_train_data:
            stu, test_id, true_score = i["user_id"], i["item_id"], i["score"]
            self.R[stu, test_id] = true_score
        self.train(epoch, epsilon)
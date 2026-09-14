# coding: utf-8
# patched EM-DINA for sparse data stability

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
    # 比 0.2 更保守一点，减少第一轮塌缩
    slip = np.zeros(shape=prob_num, dtype=np.float64) + 0.2
    guess = np.zeros(shape=prob_num, dtype=np.float64) + 0.2
    theta = np.zeros(shape=stu_num, dtype=np.int64)
    return theta, slip, guess


def safe_clip_prob(x, low=PROB_MIN, high=PROB_MAX):
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.5, posinf=high, neginf=low)
    return np.clip(x, low, high)


class DINA(CDM):
    def __init__(self, R, q_m, stu_num, prob_num, know_num, skip_value=-1):
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

    def train(self, epoch, epsilon) -> ...:
        like = np.zeros((self.stu_num, self.state_num), dtype=np.float64)
        post = np.zeros((self.stu_num, self.state_num), dtype=np.float64)

        theta = np.copy(self.theta)
        slip = safe_clip_prob(np.copy(self.slip))
        guess = safe_clip_prob(np.copy(self.guess))

        obs_mask = (self.R != self.skip_value).astype(np.float64)
        tmp_R = np.copy(self.R)
        tmp_R[self.R == self.skip_value] = 0.0

        # Beta prior smoothing，防止参数贴边
        # 数值越大，约束越强
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
            post = np.nan_to_num(post, nan=1.0 / self.state_num, posinf=1.0 / self.state_num, neginf=1.0 / self.state_num)
            post = np.maximum(post, EPS)
            post = post / np.maximum(np.sum(post, axis=1, keepdims=True), EPS)

            # M-step
            r_jl = np.dot(post.T, tmp_R)       # 正确作答的期望计数
            n_jl = np.dot(post.T, obs_mask)    # 有作答的期望计数

            r_jl_0 = np.sum(r_jl * (1 - self.eta), axis=0)
            r_jl_1 = np.sum(r_jl * self.eta, axis=0)

            n_jl_0 = np.sum(n_jl * (1 - self.eta), axis=0)
            n_jl_1 = np.sum(n_jl * self.eta, axis=0)

            n_jl_0 = np.maximum(np.nan_to_num(n_jl_0, nan=EPS, posinf=EPS, neginf=EPS), EPS)
            n_jl_1 = np.maximum(np.nan_to_num(n_jl_1, nan=EPS, posinf=EPS, neginf=EPS), EPS)

            # 带 Beta 平滑的更新
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

    def eval(self, test_data) -> tuple:
        slip = safe_clip_prob(self.slip)
        guess = safe_clip_prob(self.guess)
        pred_score = (1 - slip) * self.eta + guess * (1 - self.eta)
        pred_score = safe_clip_prob(pred_score)

        test_rmse, test_mae = [], []
        for i in tqdm(test_data, "evaluating"):
            stu, test_id, true_score = i["user_id"], i["item_id"], i["score"]
            pred = pred_score[self.theta[stu], test_id]
            pred = float(np.nan_to_num(pred, nan=0.5, posinf=0.999, neginf=0.001))
            test_rmse.append((pred - true_score) ** 2)
            test_mae.append(abs(pred - true_score))
        return np.sqrt(np.average(test_rmse)), np.average(test_mae)

    def save(self, filepath):
        with open(filepath, "wb") as file:
            pickle.dump({"theta": self.theta, "slip": self.slip, "guess": self.guess}, file)
            logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        with open(filepath, "rb") as file:
            params = pickle.load(file)
            self.theta = params["theta"]
            self.slip = safe_clip_prob(params["slip"])
            self.guess = safe_clip_prob(params["guess"])
            logging.info("load parameters from %s" % filepath)

    def inc_train(self, inc_train_data, epoch, epsilon):
        for i in inc_train_data:
            stu, test_id, true_score = i["user_id"], i["item_id"], i["score"]
            self.R[stu, test_id] = true_score
        self.train(epoch, epsilon)

    def transform(self, records):
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
import numpy as np


class Market:
    def __init__(
        self,
        num_cand: int,
        num_job: int,
        pref_cand: np.ndarray | None = None,
        pref_job: np.ndarray | None = None,
        pref_seed: int | None = None,
        pref_lambda: float = 0.8,
        rec_cand_slot: int | None = None,
        rec_job_slot: int | None = None,
        exam_cand: np.ndarray | None = None,
        exam_job: np.ndarray | None = None,
        exam_cand_type: str = "inv",
        exam_job_type: str = "inv",
        pref_structure_type: str | None = None,
    ):
        """Initialize the market with candidates, jobs, preferences, and examination probabilities.

        Args:
            num_cand (int): Number of candidates.
            num_job (int): Number of jobs.
            pref_cand (np.ndarray | None): Candidate preferences. If None, preferences are generated. Shape should be (num_cand, num_job).
            pref_job (np.ndarray | None): Job preferences. If None, preferences are generated. Shape should be (num_job, num_cand).
            pref_seed (int | None): Seed for random preference generation.
            pref_lambda (float): Lambda parameter for preference generation.
            rec_cand_slot (int | None): Number of recommendation slots for candidates. If None, it defaults to the number of jobs.
            rec_job_slot (int | None): Number of recommendation slots for jobs. If None, it defaults to the number of candidates.
            exam_cand (np.ndarray | None): Examination probabilities for candidates. If None, they are generated based on exam_cand_type. Shape should be (num_cand, rec_cand_slot).
            exam_job (np.ndarray | None): Examination probabilities for jobs. If None, they are generated based on exam_job_type. Shape should be (num_job, rec_job_slot).
            exam_cand_type (str): Examination type for candidates. Options include "log", "inv", and "exp".
            exam_job_type (str): Examination type for jobs. Options include "log", "inv", and "exp".
            pref_structure_type (str | None): Preference structure type.
                - If None, preferences are generated independently.
                - If "similar", job preferences are similar to candidate preferences.
                - If "reverse", job preferences are the reverse of candidate preferences.
        """
        self.num_cand: int = num_cand
        self.num_job: int = num_job

        if pref_seed is not None:
            np.random.seed(pref_seed)

        if pref_cand is None:
            if pref_structure_type is None:
                self.pref_cand: np.ndarray = self.__generate_pref(
                    self.num_cand, self.num_job, pref_lambda=pref_lambda
                )
            else:
                self.pref_cand: np.ndarray = self.__generate_pref(
                    self.num_cand, self.num_job, pref_lambda=0.0
                )
        else:
            self.pref_cand: np.ndarray = pref_cand.copy()

        if pref_job is None:
            if pref_structure_type is None:
                self.pref_job: np.ndarray = self.__generate_pref(
                    self.num_job, self.num_cand, pref_lambda=pref_lambda
                )
            else:
                self.pref_job: np.ndarray = self.__generate_pref(
                    self.num_job,
                    self.num_cand,
                    type=pref_structure_type,
                    opposite_pref=self.pref_cand,
                )
        else:
            self.pref_job: np.ndarray = pref_job.copy()

        if rec_cand_slot is None:
            self.rec_cand_slot: int = self.num_job
        else:
            self.rec_cand_slot: int = rec_cand_slot

        self.exam_cand_type: str | None = None
        if exam_cand is None:
            self.exam_cand_type = exam_cand_type
            self.exam_cand: np.ndarray = self.__generate_exam(
                self.num_cand, self.rec_cand_slot, self.exam_cand_type
            )
        else:
            self.exam_cand: np.ndarray = exam_cand.copy()

        if rec_job_slot is None:
            self.rec_job_slot: int = self.num_cand
        else:
            self.rec_job_slot: int = rec_job_slot

        self.exam_job_type: str | None = None
        if exam_job is None:
            self.exam_job_type = exam_job_type
            self.exam_job: np.ndarray = self.__generate_exam(
                self.num_job, self.rec_job_slot, self.exam_job_type
            )
        else:
            self.exam_job: np.ndarray = exam_job.copy()

    def __generate_pref(
        self,
        num_row: int,
        num_column: int,
        pref_lambda: float = 0.8,
        type: str | None = None,
        opposite_pref: np.ndarray | None = None,
    ) -> np.ndarray:
        if type is None:
            rand = np.random.random(size=(num_row, num_column))
            pop = np.tile(np.linspace(1, 0, num_column), (num_row, 1))
            res = (1.0 - pref_lambda) * rand + pref_lambda * pop
        elif type == "similar":
            res = np.clip(
                opposite_pref.T
                + np.random.normal(loc=0.0, scale=0.2, size=opposite_pref.T.shape),
                0.0,
                1.0,
            )
        elif type == "reverse":
            res = np.clip(
                (1.0 - opposite_pref.T)
                + np.random.normal(loc=0.0, scale=0.2, size=opposite_pref.T.shape),
                0.0,
                1.0,
            )
        else:
            raise ValueError("pref_structure_type must be 'similar' or 'reverse'!")
        return res

    def __generate_exam(
        self, num_user: int, rec_slot: int, exam_type: str
    ) -> np.ndarray:
        if exam_type == "log":
            return np.tile(1 / np.log2(np.arange(1, rec_slot + 1) + 1), (num_user, 1))
        elif exam_type == "inv":
            return np.tile(1 / np.arange(1, rec_slot + 1), (num_user, 1))
        elif exam_type == "exp":
            return np.tile(np.exp(-(np.arange(1, rec_slot + 1) - 1)), (num_user, 1))
        else:
            raise ValueError(f"Invalid exam type: {exam_type}")

    def compute_match_probability(self, rec: np.ndarray) -> np.ndarray:
        """Compute the match probabilities for each candidate-job pair based on the recommendation policy.

        Args:
            rec (np.ndarray): The recommendation policy. It can be either a deterministic recommendation of shape (num_cand, rec_cand_slot) or a stochastic recommendation of shape (num_cand, num_job, rec_cand_slot).

        Returns:
            np.ndarray: The match probability matrix of shape (num_cand, num_job).
        """
        if rec.shape == (self.num_cand, self.num_job, self.rec_cand_slot):
            return self.__compute_match_probability_stochastic(rec)
        elif rec.shape == (self.num_cand, self.rec_cand_slot):
            return self.__compute_match_probability_deterministic(rec)
        else:
            raise ValueError(f"Invalid recommendation shape: {rec.shape}")

    def __compute_match_probability_deterministic(
        self, deterministic_rec: np.ndarray
    ) -> np.ndarray:
        match_probability = np.zeros(shape=(self.num_cand, self.num_job))
        candidates = [[] for _ in range(self.num_job)]
        for i in range(self.num_cand):
            for k in range(self.rec_cand_slot):
                candidates[deterministic_rec[i, k]].append((i, k))
        for j in range(self.num_job):
            application_dist = np.zeros(self.rec_job_slot)
            application_dist[0] = 1.0
            candidates[j].sort(key=lambda x: self.pref_job[j, x[0]], reverse=True)
            for i, k in candidates[j]:
                apply_prob = self.pref_cand[i, j] * self.exam_cand[i, k]
                match_probability[i, j] = (
                    apply_prob
                    * self.pref_job[j, i]
                    * (self.exam_job[j] @ application_dist)
                )
                application_dist = (
                    1 - apply_prob
                ) * application_dist + apply_prob * np.concatenate(
                    [np.zeros(1), application_dist[:-1]]
                )
        return match_probability

    def __compute_match_probability_stochastic(
        self, stochastic_rec: np.ndarray
    ) -> np.ndarray:
        match_probability = np.zeros(shape=(self.num_cand, self.num_job))
        candidates = [i for i in range(self.num_cand)]
        for j in range(self.num_job):
            application_dist = np.zeros(self.rec_job_slot)
            application_dist[0] = 1.0
            candidates.sort(key=lambda i: self.pref_job[j, i], reverse=True)
            for i in candidates:
                apply_prob = self.pref_cand[i, j] * (
                    self.exam_cand[i] @ stochastic_rec[i, j]
                )
                match_probability[i, j] = (
                    apply_prob
                    * self.pref_job[j, i]
                    * (self.exam_job[j] @ application_dist)
                )
                application_dist = (
                    1 - apply_prob
                ) * application_dist + apply_prob * np.concatenate(
                    [np.zeros(1), application_dist[:-1]]
                )
        return match_probability

    def compute_direct_effect_suboptimality(
        self,
        rec: np.ndarray,
        match_probability: np.ndarray | None = None,
        eps: float = 0.0,
    ) -> np.ndarray:
        """Compute the suboptimality in direct effects for each candidate given a recommendation policy.

        Args:
            rec (np.ndarray): The recommendation policy. It can be either a deterministic recommendation of shape (num_cand, rec_cand_slot) or a stochastic recommendation of shape (num_cand, num_job, rec_cand_slot).
            match_probability (np.ndarray | None): The precomputed match probability matrix. If None, it will be computed using the provided recommendation policy.
            eps (float): A small threshold to consider suboptimality as zero.

        Returns:
            np.ndarray: An array of shape (num_cand,) representing the suboptimality in direct effects for each candidate.
        """
        if match_probability is None:
            match_probability = self.compute_match_probability(rec)
        candidates = [i for i in range(self.num_cand)]
        utility = np.zeros((self.num_cand, self.num_job))
        if rec.shape == (self.num_cand, self.num_job, self.rec_cand_slot):
            for j in range(self.num_job):
                application_dist = np.zeros(self.rec_job_slot)
                application_dist[0] = 1.0
                candidates.sort(key=lambda i: self.pref_job[j, i], reverse=True)
                for i in candidates:
                    utility[i, j] = (
                        self.pref_cand[i, j]
                        * self.pref_job[j, i]
                        * (application_dist @ self.exam_job[j])
                    )
                    apply_prob = self.pref_cand[i, j] * (rec[i, j] @ self.exam_cand[i])
                    application_dist = (
                        1 - apply_prob
                    ) * application_dist + apply_prob * np.concatenate(
                        [np.zeros(1), application_dist[:-1]]
                    )
        elif rec.shape == (self.num_cand, self.rec_cand_slot):
            tmp = -np.ones(shape=(self.num_cand, self.num_job), dtype=int)
            for i in range(self.num_cand):
                for k in range(self.rec_cand_slot):
                    tmp[i][rec[i, k]] = k
            for j in range(self.num_job):
                application_dist = np.zeros(self.rec_job_slot)
                application_dist[0] = 1.0
                candidates.sort(key=lambda x: self.pref_job[j, x], reverse=True)
                for i in candidates:
                    utility[i, j] = (
                        self.pref_cand[i, j]
                        * self.pref_job[j, i]
                        * (application_dist @ self.exam_job[j])
                    )
                    if tmp[i][j] >= 0:
                        apply_prob = self.pref_cand[i, j] * self.exam_cand[i, tmp[i][j]]
                        application_dist = (
                            1 - apply_prob
                        ) * application_dist + apply_prob * np.concatenate(
                            [np.zeros(1), application_dist[:-1]]
                        )
        else:
            raise ValueError(f"Invalid initial recommendation shape: {rec.shape}")

        new_rec = np.zeros(shape=(self.num_cand, self.rec_cand_slot), dtype=int)
        jobs = [j for j in range(self.num_job)]
        for i in range(self.num_cand):
            jobs.sort(key=lambda j: utility[i, j], reverse=True)
            for k in range(self.rec_cand_slot):
                new_rec[i, k] = jobs[k]
        new_match_probability = self.compute_match_probability(new_rec)
        res = new_match_probability.sum(axis=1) - match_probability.sum(axis=1)
        res[res < eps] = 0.0
        return res

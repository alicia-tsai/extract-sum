import numpy as np
import time
import scipy.sparse as sps


# Enable matlab python engine
try:
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.cd("../SMRS_v1.0")
except ImportError:
    print("Matlab not imported")

def convert_ndarray_to_matlab_mat(A):
    return matlab.double(A.tolist(), A.shape)

class Data_Reducer:
    '''
    The identity data reducer
    '''
    # def __init__(self, do_split_by_classes, do_centering, do_add_equality_constraint, term_thres, exemp_thres, num_exemp, verbose, do_SAFE, decimate):
    #     self.do_split_by_classes = do_split_by_classes
    #     self.do_centering = do_centering
    #     self.do_add_equality_constraint = do_add_equality_constraint
    #     self.term_thres = term_thres
    #     self.exemp_thres = exemp_thres
    #     self.num_exemp = num_exemp
    #     self.verbose = verbose
    #     self.do_SAFE = do_SAFE
    #     self.decimate = decimate
    #     if self.decimate is None:
    #         self.decimate = int(0.01 * self.max_iterations)
    #     self.timer = Timer(verbose = self.verbose)
    #     print("Extraneous parameters: " + str(my_params), len(my_params) > 0)
    def __init__(self, do_split_by_classes = True, credit= False, num_exemp = 100, **my_params):
        self.do_split_by_classes = do_split_by_classes
        self.credit = credit
        self.num_exemp = num_exemp
        return

    def reduce_data_credit(self, A, labels = None, A_to_reduce = None):
        '''
        A - each row corresponds to a data point
        returns a reduced data set
        '''
        assert labels is not None

        if A_to_reduce is None:
            A_to_reduce = A
        else:
            assert A.shape[0] == A_to_reduce.shape[0]


        lst_reduced_A = []
        lst_reduced_labels = []
        for label in np.unique(labels, axis=0):
            subset_A = A[labels == label]
            subset_A_to_reduce = A_to_reduce[labels == label]
            subset_labels = labels[labels == label]
            if label == 1:
                lst_reduced_A.append(subset_A_to_reduce)
                lst_reduced_labels.append(subset_labels)
            else:
                example_indicies, _ = self.identify_exemplars(subset_A)
                assert len(example_indicies) > 0, "Data reducer found 0 exemplars"

                lst_reduced_A.append(subset_A_to_reduce[example_indicies])
                lst_reduced_labels.append(subset_labels[example_indicies])
        reduced_A = np.vstack(lst_reduced_A)
        reduced_labels = np.concatenate(lst_reduced_labels)
        return reduced_A, reduced_labels

    def reduce_data(self, A, labels = None, A_to_reduce = None):
        '''
        A - each row corresponds to a data point
        returns a reduced data set
        '''
        if A_to_reduce is None:
            A_to_reduce = A
        else:
            assert A.shape[0] == A_to_reduce.shape[0]

        if self.credit:
            return self.reduce_data_credit(A, labels)

        if labels is None or not self.do_split_by_classes:
            example_indicies, _ = self.identify_exemplars(A)
            assert len(example_indicies) > 0, "Data reducer found 0 exemplars"
            if labels is None:
                return A_to_reduce[example_indicies]
            return A_to_reduce[example_indicies], labels[example_indicies]

        lst_reduced_A = []
        lst_reduced_labels = []
        for label in np.unique(labels, axis=0):
            subset_A = A[labels == label]
            subset_A_to_reduce = A_to_reduce[labels == label]
            subset_labels = labels[labels == label]

            example_indicies, _ = self.identify_exemplars(subset_A)
            assert len(example_indicies) > 0, "Data reducer found 0 exemplars"

            lst_reduced_A.append(subset_A_to_reduce[example_indicies])
            lst_reduced_labels.append(subset_labels[example_indicies])
        reduced_A = np.vstack(lst_reduced_A)
        reduced_labels = np.concatenate(lst_reduced_labels)
        return reduced_A, reduced_labels

    def make_exemplar_indices(self, Z, num_exemp):
        """
        horizontal_norms refers to the horizontal norms of ZT which are the vertical norms of Z
        """
        horizontal_norms = np.linalg.norm(Z, ord=2, axis = 0)
        total_norm_sum = np.sum(horizontal_norms)
        sorted_indices = np.flipud(np.argsort(horizontal_norms))[:num_exemp]

        m = Z.shape[0]

        #don't pick coefficients that aren't used at all
        last_index = num_exemp
        for idx in range(len(sorted_indices)):
            og_idx = sorted_indices[idx]
            if horizontal_norms[og_idx] == 0.0:
                last_index = idx
                #print("ALERT: less than num_exemp were selected")
                break

        return sorted_indices[:last_index]

    def make_exemplar_indices_1(self, Z, num_exemp):
        horizontal_norms = np.linalg.norm(Z, ord=2, axis = 0)
        sorted_indices = np.flipud(np.argsort(horizontal_norms))

        cur_norm_sum = 0
        total_norm_sum = np.sum(horizontal_norms)
        for j in range(len(sorted_indices)):
            index = sorted_indices[j]
            cur_norm_sum += horizontal_norms[index]
            if cur_norm_sum / total_norm_sum > self.exemp_thres:
                return sorted_indices[: j + 1]
        print("IT SHOULD NEVER GET HERE")
        return sorted_indices


class DR_SMRS_Coordinate_Descent(Data_Reducer):
    '''
    A - each row corresponds to a datapoint
    '''
    def __init__(self, alpha= None, lmbda= None, do_scale = False, zeta = 1, do_split_by_classes = True,
        do_centering = True, do_add_equality_constraint = False, term_thres = 1e-4, exemp_thres = 0.01,
        num_exemp = 100, max_iterations=10000, do_SAFE = False, verbose=True, decimate = None, positive = False, **my_params):
        if alpha is not None and lmbda is not None:
            raise Exception("alpha, " + str(alpha) + ", and lmbda, " + str(lmbda) + ", are both not None values")
        elif alpha is not None:
            self.alpha = alpha
            self.lmbda = None
        else:
            self.alpha = None
            self.lmbda = lmbda

        self.do_scale = do_scale
        self.zeta = zeta
        self.do_split_by_classes = do_split_by_classes
        self.do_centering = do_centering
        self.do_add_equality_constraint = do_add_equality_constraint
        self.term_thres = term_thres
        self.exemp_thres = exemp_thres
        self.num_exemp = num_exemp
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.do_SAFE = do_SAFE
        self.timer = Timer(verbose = False)
        self.decimate = decimate
        if self.decimate is None:
            self.decimate = int(0.01 * self.max_iterations)
        self.positive = positive
        print("Extraneous parameters: " + str(my_params), len(my_params) > 0)

    def compute_lmbda_max(self, AT):
        """Refer to Vidal's code to compute lmbda max. This is essentially computing the largest value the"""
        A = AT.T
        return eng.computeLambda_mat(convert_ndarray_to_matlab_mat(A), False)

    def reduce_data_credit(self, AT, labels = None):
        '''Returns a reduced data set'''
        #Create lmbda if alpha is not None
        assert labels is not None
        is_sparse = sps.issparse(AT)

        if self.alpha is not None:
            self.lmbda = self.compute_lmbda_max(AT) / self.alpha

        lst_reduced_AT = []
        lst_reduced_labels = []
        for label in np.unique(labels, axis=0):
            print("", self.verbose)
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$", self.verbose)
            print("Label: " + str(label), self.verbose)

            subset_AT = AT[labels == label]
            if is_sparse:
                og_subset_AT = subset_AT.copy()
            else:
                og_subset_AT = np.copy(subset_AT)
            subset_labels = labels[labels == label]

            if label == 1:
                lst_reduced_AT.append(og_subset_AT)
                lst_reduced_labels.append(subset_labels)
            else:
                if self.do_centering:
                    subset_AT = self.center_data(subset_AT)

                if self.do_scale:
                    AT = sklearn.preprocessing.scale(AT, axis=0, with_mean = True, with_std=True, copy=True)

                if self.do_add_equality_constraint:
                    subset_AT = self.add_equality_constraint(subset_AT, self.zeta)

                if self.do_SAFE:
                    kept_data_indices = self.run_SAFE(subset_AT, self.lmbda)
                    subset2_AT = subset_AT[kept_data_indices]

                    exemplar_indices, _ = self.identify_exemplars(subset2_AT)
                    final_indices = kept_data_indices[exemplar_indices]
                else:
                    final_indices, _ = self.identify_exemplars(subset_AT)

                assert len(final_indices) > 0, "Data reducer found 0 exemplars"

                lst_reduced_AT.append(og_subset_AT[final_indices])
                lst_reduced_labels.append(subset_labels[final_indices])

        if is_sparse:
            reduced_AT = sps.vstack(lst_reduced_AT)
        else:
            reduced_AT = np.vstack(lst_reduced_AT)
        reduced_labels = np.concatenate(lst_reduced_labels)

        #reset lmbda alpha is being used
        if self.alpha is not None:
            self.lmbda = None
        return reduced_AT, reduced_labels

    def reduce_data(self, AT, labels = None):
        '''Returns a reduced data set'''
        if self.credit:
            return self.reduce_data_credit(AT, labels)

        is_sparse = sps.issparse(AT)

        if labels is None or not self.do_split_by_classes:
            if self.alpha is not None:
                self.lmbda = self.compute_lmbda_max(AT) / self.alpha

            if is_sparse:
                og_AT = AT.copy()
            else:
                og_AT = np.copy(AT)
            if self.do_centering:
                AT = self.center_data(AT)

            if self.do_scale:
                AT = sklearn.preprocessing.scale(AT, axis=0, with_mean = True, with_std=True, copy=True)

            if self.do_add_equality_constraint:
                AT = self.add_equality_constraint(AT, self.zeta)

            if self.do_SAFE:
                kept_data_indices = self.run_SAFE(AT, self.lmbda)
                subset2_AT = AT[kept_data_indices]

                exemplar_indices, extra = self.identify_exemplars(subset2_AT)
                final_indices = kept_data_indices[exemplar_indices]
            else:
                final_indices, extra = self.identify_exemplars(AT)

            new_AT = og_AT[final_indices]
            if self.reweight:
                X = extra[0]
                weights = np.linalg.norm(X, axis = 1, ord = 2)
                weights = weights[final_indices].reshape((-1, 1))
                new_AT *= weights


            assert len(final_indices) > 0, "Data reducer found 0 exemplars"
            if labels is None:
                return new_AT
            return new_AT, labels[final_indices]

        lst_reduced_AT = []
        lst_reduced_labels = []
        for label in np.unique(labels, axis=0):
            print("", self.verbose)
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$", self.verbose)
            print("Label: " + str(label), self.verbose)
            subset_AT = AT[labels == label]
            if is_sparse:
                og_subset_AT = subset_AT.copy()
            else:
                og_subset_AT = np.copy(subset_AT)
            subset_labels = labels[labels == label]

            if self.alpha is not None:
                self.lmbda = self.compute_lmbda_max(og_subset_AT) / self.alpha

            if self.do_centering:
                subset_AT = self.center_data(subset_AT)

            if self.do_scale:
                subset_AT = sklearn.preprocessing.scale(subset_AT, axis=0, with_mean = True, with_std=True, copy=True)

            if self.do_add_equality_constraint:
                subset_AT = self.add_equality_constraint(subset_AT, self.zeta)

            if self.do_SAFE:
                kept_data_indices = self.run_SAFE(subset_AT, self.lmbda)
                subset2_AT = subset_AT[kept_data_indices]

                exemplar_indices, extra = self.identify_exemplars(subset2_AT)
                final_indices = kept_data_indices[exemplar_indices]
            else:
                final_indices, extra = self.identify_exemplars(subset_AT)
                print(final_indices[0])
                print(len(final_indices))


            assert len(final_indices) > 0, "Data reducer found 0 exemplars"
            new_AT = og_subset_AT[final_indices]
            if self.reweight:
                X = extra[0]
                weights = np.linalg.norm(X, axis = 1, ord = 2)
                weights = weights[final_indices].reshape((-1, 1))
                new_AT *= weights

            lst_reduced_AT.append(new_AT)
            lst_reduced_labels.append(subset_labels[final_indices])

        if is_sparse:
            reduced_AT = sps.vstack(lst_reduced_AT)
        else:
            reduced_AT = np.vstack(lst_reduced_AT)
        reduced_labels = np.concatenate(lst_reduced_labels)

        #reset lmbda alpha is being used
        if self.alpha is not None:
            self.lmbda = None
        return reduced_AT, reduced_labels

    def add_equality_constraint(self, AT, zeta):
        '''
        Adds the 1^T Z^T = 1^T constraint to penalty function
        '''
        print("Adding equality constraint", self.verbose)
        return np.hstack((np.ones([AT.shape[0], 1]) * zeta, AT))

    def center_data(self, AT):
        '''
        AT - m x n; n dimensions m data points
        '''
        # A = AT.T
        # ("Centering data", self.verbose)
        # center = np.mean(A, axis=1).reshape((A.shape[0], 1))
        # tiled_center = np.tile(center, A.shape[1])
        # assert tiled_center.shape == A.shape, str(center.shape) + str(A.shape)
        # zero_mean_data = A - tiled_center
        # return zero_mean_data.T

        A = AT.T
        print("Centering data", self.verbose)
        center = np.mean(A, axis=0).reshape((1, -1))
        tiled_center = np.tile(center, (A.shape[0], 1))
        assert tiled_center.shape == A.shape, str(center.shape) + str(A.shape)
        zero_mean_data = A - tiled_center
        return zero_mean_data.T

    def run_SAFE(self, subset_A, lmbda):
        ret_val = np.where(np.linalg.norm(subset_A, ord=2, axis=1) < lmbda)[0]
        num_data_pts_eliminated = subset_A.shape[0] - len(ret_val)
        frac_data_pts_eliminated = num_data_pts_eliminated / subset_A.shape[0]
        print("########## SAFE ##########", self.verbose)
        print("Num data eliminated: " + str(num_data_pts_eliminated), self.verbose)
        print("Fraction of data eliminated: " + str(frac_data_pts_eliminated), self.verbose)
        return ret_val

    ### HELPER
    def objective(self, V, Z, lmbda):
        return np.linalg.norm(V, ord="fro") ** 2  + lmbda * np.sum(np.linalg.norm(Z, ord=2, axis=0))
    def og_objective(self, A, Z, lmbda):
        return np.linalg.norm(A - A.dot(Z.T), ord="fro") ** 2  + lmbda * np.sum(np.linalg.norm(Z, ord=2, axis=0))


class DR_Frank_Wolfe(DR_SMRS_Coordinate_Descent):
    def __init__(self, init_proportion = None, linear = False, reweight = False, do_scale = False, epsilon = 0, credit = False,
    alpha = None, beta = None, zeta = 1, positive =False, greedy = True, order = 2, gamma = None, do_rbf_kernel = False, 
    do_split_by_classes = True, do_centering = True, do_add_equality_constraint = False, term_thres = 1e-8, exemp_thres = 0.01,
    num_exemp = 100, max_iterations=10000, do_SAFE = False, verbose=True, decimate = None, **my_params):
        self.alpha = alpha
        self.beta = beta

        assert self.beta is None or self.alpha is None

        self.lmbda = None
        self.init_proportion = init_proportion
        self.linear = linear
        self.reweight = reweight
        self.do_scale = do_scale
        self.epsilon = epsilon
        self.credit = credit
        self.zeta = zeta
        self.greedy = greedy
        self.positive = positive
        self.order = order #specifies which ord is used for the inner norm in the group lasso ball
        self.do_rbf_kernel = do_rbf_kernel
        self.gamma = gamma
        self.do_split_by_classes = do_split_by_classes
        self.do_centering = do_centering
        self.do_add_equality_constraint = do_add_equality_constraint
        self.term_thres = term_thres
        self.exemp_thres = exemp_thres
        self.num_exemp = num_exemp
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.do_SAFE = do_SAFE
        self.timer = Timer(verbose = False)
        self.decimate = decimate
        if self.decimate is None:
            self.decimate = int(0.01 * self.max_iterations)

    def get_max_index(self, gradient, order, positive):
        if positive:
            if np.all(gradient >= 0):
                return -1
            gradient = np.where(gradient < 0 , gradient, 0)

        if order == 2:
            return np.argmax(np.linalg.norm(gradient, axis=1, ord=2))
        elif np.isinf(order):
            return np.argmax(np.linalg.norm(gradient, axis=1, ord=1))
        elif order == 1:
            return np.argmax(np.linalg.norm(gradient, axis=1, ord=np.inf))
        raise Exception("Improper ord arguement; ord = " + str(ord))

    def make_S_row(self, gradient_max_row, beta, order, positive):
        if positive:
            return self.make_S_row_positive(gradient_max_row, beta, order)

        if order == 2:
            if np.linalg.norm(gradient_max_row, ord=2) == 0:
                val = np.zeros_like(gradient_max_row)
                val[0] = beta
                return val
            return -1 * gradient_max_row / np.linalg.norm(gradient_max_row, ord=2) * beta + 0.
        if np.isinf(order):
            sign_vec = np.sign(gradient_max_row)
            sign_vec[sign_vec == 0] = 1 #this is just to make sure a vertex of the ball is selected
            return -1 * sign_vec * beta + 0.
        if order == 1:
            max_index = np.argmax(np.abs(gradient_max_row))
            max_sign = np.sign(gradient_max_row[max_index])

            if max_sign == 0:
                max_sign = 1 #this is just to make sure a vertex of the ball is selected

            return_vec = np.zeros_like(gradient_max_row)
            return_vec[max_index] = -1 * max_sign * beta
            return return_vec + 0.

    def make_S_row_positive(self, gradient_max_row, beta, order):
        gradient_max_row = np.where(gradient_max_row < 0 , gradient_max_row, 0.)
        if order == 2:
            return -1 * gradient_max_row / np.linalg.norm(gradient_max_row, ord=2) * beta + 0.
        if np.isinf(order):
            sign_vec = np.sign(gradient_max_row)
            return -1 * sign_vec * beta + 0.
        if order == 1:
            max_index = np.argmax(np.abs(gradient_max_row))
            max_sign = np.sign(gradient_max_row[max_index])

            return_vec = np.zeros_like(gradient_max_row)
            return_vec[max_index] = -1 * max_sign * beta
            return return_vec + 0.

    def compute_inner_product_of_S_max_row(self, m, beta, order):
        """To compute the optimal step size, one of the trace terms need the inner product of s_max^T s_max
        This calculation depends on the order of the group lasso ball.
        """
        if order == 2 or order == 1:
            return beta ** 2
        elif np.isinf(order):
            return (beta ** 2) * m

    def add_equality_constraint(self, AT, zeta):
        '''
        We want to add equality constraint in kernel space not in normal space
        '''

        return AT # I'm not appending the 1's

        if self.do_rbf_kernel:
            return AT
        print("Adding equality constraint", self.verbose)
        if sps.issparse(AT):
            my_ones = sps.csr_matrix(np.ones([AT.shape[0], 1]) * zeta)
            return sps.hstack((my_ones, AT))
        return np.hstack((np.ones([AT.shape[0], 1]) * zeta, AT))

    def identify_exemplars(self, subset_A, K = None, warm_X = None):
        """This makes it more expensive to use K directly when doing A^T A X, so we avoid it by using AX more."""
        set_beta_to_None = False
        if self.beta is None and self.alpha is not None:
            self.beta = subset_A.shape[0] / self.alpha
            set_beta_to_None = True
        elif self.linear:
            self.beta = 2* subset_A.shape[0]
        elif self.alpha is None and self.beta is None:
            raise Exception("beta, " + str(beta) + ", cannot be a None value")

        K_is_None = K is None
        if sps.issparse(subset_A):
            #print("A is sparse")
            return self.identify_exemplars_sparse(subset_A)

        self.timer.start()
        A = subset_A.T

        if K_is_None and self.do_rbf_kernel:
            K = rbf_kernel(A.T, A.T, gamma=self.gamma)
        elif K_is_None:
            K = A.T.dot(A)
        if self.do_add_equality_constraint: #and self.do_rbf_kernel:
            K += self.zeta ** 2

        #print(type(K))  #TODO: BUG
        trK = np.trace(K)
        n, m = A.shape

        if warm_X is not None:
            X = warm_X
        else:
            X = np.zeros((m, m))

        #Greedily select first
        #exemplar_index_set = {}
        exemplar_index_lst = []
        iteration = 0
        cost_lst = []
        G_lst = []

        # L = 2 * np.linalg.norm(K, ord="fro")

        row_norm_X = np.linalg.norm(X, axis = 1, ord = 2)
        exemplar_index_lst = np.where(row_norm_X != 0)[0]
        len_of_exemplar_index_lst = [len(exemplar_index_lst)]

        #print(self.beta)
        #pbar = tqdm(total = int(self.max_iterations), unit="iter", unit_scale=False, leave=False)
        try:
            #pdb.set_trace()
            for iteration in range(int(self.max_iterations)):
                if self.greedy and len(exemplar_index_lst) >= self.num_exemp:
                    #pbar.update(int(self.max_iterations) - iteration)
                    break
                #pbar.set_postfix(num_exemplars = len_of_exemplar_index_lst[-1], refresh=False)
                #pbar.update(1)

                if False and n < len(exemplar_index_lst) and n < m and not self.do_rbf_kernel:
                # Because I no longer append row of 1's when adding equality constraint, we need to always use the K matrix
                    # in this situation we want to avoid the explicity use of K
                    AX = A.dot(X)
                    KX = A.T.dot(AX)
                    trXTK = np.trace(KX)
                    trXTKX = np.linalg.norm(AX, ord="fro")**2
                else:
                    #in this situation, we would rather use K than A
                    if len(exemplar_index_lst) == 0:
                        KX = np.zeros((m, m))
                    else:
                        KX = K[:, exemplar_index_lst].dot(X[exemplar_index_lst])
                    trXTK = np.trace(KX)
                    trXTKX = np.einsum('ij,ji->', X.T, KX)


                cost_lst.append(trXTKX - 2 * trXTK + trK)  # ||AX - A||_F: forbenius norm

                # Gradient calculation
                if self.epsilon == 0:
                    gradient = 2 * (KX - K)#with respect to Z
                else:
                    gradient = 2 * (KX - K + self.epsilon * X) #with respect to Z

                max_index = self.get_max_index(gradient=gradient, order=self.order, positive = self.positive) #next index to update
                if max_index == -1 and self.positive:
                    S = np.zeros((m, m))
                    D = -X

                    trXTK = np.trace(KX)
                    numerator = - trXTK + trXTKX
                    denominator = trXTKX
                else:
                    gradient_max_row = gradient[max_index].flatten()
                    S = np.zeros((m, m))
                    S[max_index] = self.make_S_row(gradient_max_row=gradient_max_row, beta=self.beta, order=self.order , positive = self.positive)
                    D = S - X

                    ### NEW2: even more efficient
                    trSTK = np.inner(S[max_index], K[max_index])
                    trSTKS = K[max_index, max_index] * np.inner(S[max_index], S[max_index])
                    trSTKX = np.inner(S[max_index], KX[max_index])

                    numerator = trSTK - trXTK - trSTKX + trXTKX
                    denominator = trSTKS - 2 * trSTKX + trXTKX

                #Early termination if there are less than num_exemp exemplars
                G = - np.einsum("ij, ij ->", gradient, D)
                G_lst.append(G)
                if G < self.term_thres:
                    break

                if self.linear and iteration == 0:
                    step_size = 1.
                else:
                    step_size = max(0, min(1, numerator / denominator))

                X += step_size * D
                row_norm_X = np.linalg.norm(X, axis = 1, ord=2)
                exemplar_index_lst = np.where(row_norm_X != 0)[0]
                len_of_exemplar_index_lst.append(len(exemplar_index_lst))
        except KeyboardInterrupt:
            print("Graceful Interrruption")
            time.sleep(1)

        #pbar.close()
        if not self.greedy:
            exemplar_indices = self.make_exemplar_indices(X.T, self.num_exemp)
        else:
            exemplar_indices = exemplar_index_lst
            #if len(exemplar_indices) < self.num_exemp:
                #print("here? ALERT: less than num_exemp were selected: " + str(len(exemplar_indices)))
        self.timer.stop()

        if set_beta_to_None:
            self.beta = None

        return exemplar_indices, (X, len_of_exemplar_index_lst, cost_lst, G_lst)

    def fw_objective(self, AX, X):
        return np.linalg.norm(AX - A, ord="fro")**2

    def identify_exemplars_sparse(self, subset_A):
        """This makes it more expensive to use K directly when doing A^T A X, so we avoid it by using AX more.

        A is sparse.csc_matrix
        K is usually not a sparse matrix
        sparsity is primarily for the large KX matrix multiplication
        """
        self.timer.start()
        A = subset_A.T
        if not self.do_rbf_kernel:
            K = A.T.dot(A) # row format
            K_dense = K.toarray()
        else:
            K_dense = rbf_kernel(A.T, A.T, gamma=self.gamma)

        if self.do_add_equality_constraint and self.do_rbf_kernel:
            K_dense += self.zeta ** 2


        trK = np.trace(K_dense)
        n, m = A.shape
        X = sps.csr_matrix((m, m))

        #Greedily select first
        exemplar_index_set = {}
        exemplar_index_lst = []
        iteration = 0
        cost_lst = []
        G_lst = []
        while (len(exemplar_index_lst) < self.num_exemp or not self.greedy) and iteration < self.max_iterations:
            if n < m and not self.do_rbf_kernel:
                # in this situation we want to avoid the explicity use of K
                AX = A.dot(X)
                AX_dense = AX.toarray()
                KX = AX_dense.T.dot(AX_dense)  #TODO:BUG
                trXTKX = np.linalg.norm(AX_dense, ord="fro")**2
            else:
                #in this situation, we would rather use K than A
                X_dense = X.toarray()
                KX = K_dense.dot(X_dense)
                trXTKX = np.sum([np.inner(X_dense[:, i].flatten(), KX[:, i].flatten()) for i in range(m)]) #TODO #this one is a bit faster.

            cost_lst.append(trXTKX - 2 * np.trace(KX) + trK)
            gradient = 2 * KX - 2 * K_dense  #with respect to Z

            max_index = self.get_max_index(gradient=gradient, order=self.order, positive = self.positive) #next index to update
            if max_index is None and self.positive:
                S = sps.csr_matrix((m, m))
                D = -X

                trXTK = np.trace(KX)
                numerator = - trXTK + trXTKX
                denominator = trXTKX
            else:
                gradient_max_row = gradient[max_index].flatten()
                if max_index not in exemplar_index_set:
                    exemplar_index_set[max_index] = "empty"
                    exemplar_index_lst.append(max_index)
                S = sps.lil_matrix((m, m), dtype = np.float64)
                S_max_row = np.array(self.make_S_row(gradient_max_row=gradient_max_row, beta=self.beta, order=self.order , positive = self.positive)).flatten()
                S[max_index] = S_max_row #TODO
                S = S.tocsr()
                D = S - X

                ### NEW2: even more efficient
                trSTK = np.inner(S_max_row, K_dense[max_index])
                trSTKS = K_dense[max_index, max_index] * np.inner(S_max_row, S_max_row)
                trXTK = np.trace(KX)
                trSTKX = np.inner(S_max_row, KX[max_index])

                numerator = trSTK - trXTK - trSTKX + trXTKX
                denominator = trSTKS - 2 * trSTKX + trXTKX

            #Early termination if there are less than num_exemp exemplars
            D_dense = D.toarray()
            G = - np.sum(gradient * D_dense)
            G_lst.append(G)
            if G < self.term_thres:
                break

            step_size = min(1, numerator / denominator)
            X += step_size * D
            iteration += 1

        X_dense = X.toarray()
        if not self.greedy:
            exemplar_indices = self.make_exemplar_indices(X_dense.T, self.num_exemp)
        else:
            exemplar_indices = exemplar_index_lst
            # if len(exemplar_indices) < self.num_exemp:
            #     print("ALERT: less than num_exemp were selected: " + str(len(exemplar_indices)))
        self.timer.stop()

        # print("##########################", self.verbose)
        # print("Number of Iterations: " + str(iteration + 1), self.verbose)
        # print("Duration: " + str(self.timer.get_latest()), self.verbose)
        num_exemplar = len(exemplar_indices)
        frac_exemplar = num_exemplar / A.shape[1]
        # print("Num exemplars: " + str(num_exemplar), self.verbose)
        # print("Fraction exemplars: " + str(frac_exemplar), self.verbose)

        return exemplar_indices, (X_dense, cost_lst, G_lst)


class Timer:
    def __init__(self, verbose = True):
        self.verbose = verbose
        self.is_timing = False
        self.start_time = None
        self.durations = []
        self.current_duration = 0
    def start(self, fresh=True):
        self.is_timing = True
        self.start_time = time.time()
        if fresh:
            self.current_duration = 0
    def pause(self):
        self.is_timing = False
        self.current_duration += time.time() - self.start_time
    def stop(self):
        if self.is_timing:
            self.durations.append(time.time() - self.start_time + self.current_duration)
            self.is_timing = False
            self.current_duration = 0
            #print("Duration of timing is: " + str(self.get_latest()), self.verbose)
    def clear(self):
        self.durations = []

    def get_average(self):
        return np.mean(self.durations)
    def get_latest(self):
        return self.durations[-1]

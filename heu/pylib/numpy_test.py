#  Copyright 2022 Ant Group Co., Ltd.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import pickle
import sys
import unittest

import numpy as np

from heu import numpy as hnp
from heu import phe


class BasicCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kit = hnp.setup(phe.SchemaType.Mock, 2048)
        cls.encryptor = cls.kit.encryptor()
        cls.decryptor = cls.kit.decryptor()
        cls.evaluator = cls.kit.evaluator()

    def assert_array_equal(self, harr, nparr, edr=phe.BigintDecoder()):
        if isinstance(harr, hnp.CiphertextArray):
            harr = self.decryptor.decrypt(harr)

        harr = harr.to_numpy(edr)
        self.assertTrue(
            np.array_equal(harr, nparr),
            f"hnp is\n{harr}, type {type(harr)}\nnumpy array is \n{nparr}, type {type(nparr)}",
        )

    def test_parse_and_str(self):
        harr = self.kit.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(str(harr), "[[1 2 3]\n [4 5 6]]")
        harr = hnp.array([[1, 2, 3], [4, 5, 6]], self.kit.integer_encoder(scale=10))
        self.assertEqual(str(harr), "[[10 20 30]\n [40 50 60]]")

    def test_io(self):
        def do_test(input, encoder):
            # test raw parse
            harr = hnp.array(input, encoder)
            pyarr = harr.to_numpy(encoder)
            self.assertTrue(
                np.array_equal(pyarr, np.array(input)),
                f"hnp is {pyarr}, input is {input}",
            )

            # test works with numpy ndarray
            harr = hnp.array(np.array(input), encoder)
            pyarr = harr.to_numpy(encoder)
            self.assertTrue(
                np.array_equal(pyarr, np.array(input)),
                f"hnp is {pyarr}, input is {input}",
            )

        do_test(666, phe.IntegerEncoder(self.kit.get_schema()))
        do_test(666, phe.FloatEncoder(self.kit.get_schema()))
        do_test(666, phe.BigintEncoder(self.kit.get_schema()))

        do_test([1, 2, 3], phe.IntegerEncoder(self.kit.get_schema()))
        do_test([[1, 2, 3], [4, 5, 6]], phe.IntegerEncoder(self.kit.get_schema()))
        do_test([[10], [20]], phe.IntegerEncoder(self.kit.get_schema()))

        do_test([100.1], phe.FloatEncoder(self.kit.get_schema()))
        do_test([[1.1, 2.4], [3.6, 8]], phe.FloatEncoder(self.kit.get_schema()))

        do_test([1, 2, 3], phe.BigintEncoder(self.kit.get_schema()))
        do_test([[92233720368547758070]], phe.BigintEncoder(self.kit.get_schema()))
        do_test(
            [47509577600241629199431517033180053701475],
            phe.BigintEncoder(self.kit.get_schema()),
        )
        do_test(
            [[47509577600241629199431517033180053701475], [12]],
            phe.BigintEncoder(self.kit.get_schema()),
        )

        do_test([1, 2], phe.BatchIntegerEncoder(self.kit.get_schema()))
        do_test(
            [[10, 11], [12, 13], [15, 16]],
            phe.BatchIntegerEncoder(self.kit.get_schema()),
        )

    def test_encoder_parallel(self):
        edr = self.kit.integer_encoder()
        for idx in range(10):
            input = np.random.randint(-10000, 10000, (100, 100))
            harr = self.kit.array(input, edr)
            self.assert_array_equal(harr, input, edr)

        edr = phe.FloatEncoder(self.kit.get_schema(), scale=10**8)
        for idx in range(50):
            input = np.random.rand(100, 100)
            harr = self.kit.array(input, edr)
            pyarr = harr.to_numpy(edr)
            self.assertTrue(
                np.allclose(pyarr, input), f"hnp is \n{pyarr}\ninput is \n{input}"
            )

        # BigintEncoder, this is default
        for idx in range(50):
            input = np.random.randint(-10000, 10000, (100, 100))
            harr = self.kit.array(input)
            self.assert_array_equal(harr, input)

        # batch int
        for idx in range(50):
            input = np.random.randint(-10000, 10000, (5000, 2))
            harr = self.kit.array(input, phe.BatchIntegerEncoderParams())
            self.assert_array_equal(harr, input, self.kit.batch_integer_encoder())

        # batch float
        for idx in range(50):
            input = np.random.rand(5000, 2)
            harr = self.kit.array(input, phe.BatchFloatEncoderParams(scale=2**62))
            self.assert_array_equal(
                harr, input, self.kit.batch_float_encoder(scale=2**62)
            )

    def test_encrypt_with_audit(self):
        pt1 = self.kit.array([[1], [3]])
        ct1, audit = self.encryptor.encrypt_with_audit(pt1)
        self.assert_array_equal(self.decryptor.decrypt(ct1), np.array([[1], [3]]))

        buf = pickle.dumps(audit)
        a2 = pickle.loads(buf)
        self.assertEqual(a2.ndim, 2)
        self.assertEqual(tuple(a2.shape), (2, 1))
        self.assertGreater(len(a2[0, 0]), 1)

    def test_evaluate(self):
        pt1 = self.kit.array([[1, 2], [3, 4]])
        pt2 = self.kit.array([[4, 5], [6, 7]])
        self.assertEqual(pt2.rows, 2)
        self.assertEqual(pt2.cols, 2)
        self.assertEqual(tuple(pt2.shape), (2, 2))
        ct1 = self.encryptor.encrypt(pt1)
        ct2 = self.encryptor.encrypt(pt2)
        self.assertEqual(ct2.rows, 2)
        self.assertEqual(ct2.cols, 2)
        self.assertEqual(tuple(ct2.shape), (2, 2))

        # add
        ans = np.array([[5, 7], [9, 11]])
        pt3 = self.evaluator.add(pt1, pt2)
        buf = pickle.dumps(pt3)
        self.assert_array_equal(pickle.loads(buf), ans)
        ct3 = self.evaluator.add(ct1, pt2)
        buf = pickle.dumps(ct3)
        self.assert_array_equal(self.decryptor.decrypt(pickle.loads(buf)), ans)
        ct3 = self.evaluator.add(pt1, ct2)
        self.assert_array_equal(self.decryptor.decrypt(ct3), ans)
        ct3 = self.evaluator.add(ct1, ct2)
        self.assert_array_equal(self.decryptor.decrypt(ct3), ans)

        # sub
        ans = np.array([[-3, -3], [-3, -3]])
        pt3 = self.evaluator.sub(pt1, pt2)
        buf = pickle.dumps(pt3)
        self.assert_array_equal(pickle.loads(buf), ans)
        ct3 = self.evaluator.sub(ct1, pt2)
        buf = pickle.dumps(ct3)
        self.assert_array_equal(self.decryptor.decrypt(pickle.loads(buf)), ans)
        ct3 = self.evaluator.sub(pt1, ct2)
        self.assert_array_equal(self.decryptor.decrypt(ct3), ans)
        ct3 = self.evaluator.sub(ct1, ct2)
        self.assert_array_equal(self.decryptor.decrypt(ct3), ans)

        # mul
        ans = np.array([[4, 10], [18, 28]])
        pt3 = self.evaluator.mul(pt1, pt2)
        buf = pickle.dumps(pt3)
        self.assert_array_equal(pickle.loads(buf), ans)
        ct3 = self.evaluator.mul(ct1, pt2)
        buf = pickle.dumps(ct3)
        self.assert_array_equal(self.decryptor.decrypt(pickle.loads(buf)), ans)
        ct3 = self.evaluator.mul(pt1, ct2)
        self.assert_array_equal(self.decryptor.decrypt(ct3), ans)

        # matmul
        ans = np.array([[16, 19], [36, 43]])
        pt3 = self.evaluator.matmul(pt1, pt2)
        buf = pickle.dumps(pt3)
        self.assert_array_equal(pickle.loads(buf), ans)
        ct3 = self.evaluator.matmul(ct1, pt2)
        buf = pickle.dumps(ct3)
        self.assert_array_equal(self.decryptor.decrypt(pickle.loads(buf)), ans)
        ct3 = self.evaluator.matmul(pt1, ct2)
        self.assert_array_equal(self.decryptor.decrypt(ct3), ans)

        # sum
        pt3 = self.evaluator.sum(pt1)
        self.assertEqual(pt3, phe.Plaintext(self.kit.get_schema(), 10))
        ct3 = self.evaluator.sum(ct1)
        self.assertEqual(
            self.decryptor.phe.decrypt(ct3), phe.Plaintext(self.kit.get_schema(), 10)
        )
        pt3 = self.evaluator.sum(pt2)
        self.assertEqual(pt3, phe.Plaintext(self.kit.get_schema(), 22))
        ct3 = self.evaluator.sum(ct2)
        self.assertEqual(
            self.decryptor.phe.decrypt(ct3), phe.Plaintext(self.kit.get_schema(), 22)
        )

        # select sum
        pt3 = self.evaluator.select_sum(pt1, ([0, 1], 0))
        self.assertEqual(pt3, phe.Plaintext(self.kit.get_schema(), 4))
        ct3 = self.evaluator.select_sum(ct1, [0])
        self.assertEqual(
            self.decryptor.phe.decrypt(ct3), phe.Plaintext(self.kit.get_schema(), 3)
        )
        pt3 = self.evaluator.select_sum(pt2, ())
        self.assertEqual(pt3, phe.Plaintext(self.kit.get_schema(), 0))
        ct3 = self.evaluator.select_sum(ct2, (1, [0, 1]))
        self.assertEqual(
            self.decryptor.phe.decrypt(ct3), phe.Plaintext(self.kit.get_schema(), 13)
        )

        # batch select sum
        pt3 = self.evaluator.batch_select_sum(pt1, [(0, 0), (1, 1)])
        self.assert_array_equal(pt3, np.array([1, 4]))
        ct3 = self.evaluator.batch_select_sum(ct1, [0])
        self.assert_array_equal(ct3, np.array([3]))
        pt3 = self.evaluator.batch_select_sum(pt2, [(), ()])
        self.assert_array_equal(pt3, np.array([0, 0]))
        ct3 = self.evaluator.batch_select_sum(ct2, [(1, [0, 1]), ([0, 1], 0)])
        self.assert_array_equal(ct3, np.array([13, 10]))

    def test_matmul(self):
        nparr1 = np.random.randint(-10000, 10000, (64,))
        harr1 = self.kit.array(nparr1)
        nparr2 = np.random.randint(-10000, 10000, (64, 256))
        harr2 = self.kit.array(nparr2)
        self.assert_array_equal(self.evaluator.matmul(harr1, harr2), nparr1 @ nparr2)

        harr2 = harr2.transpose()
        self.assert_array_equal(self.evaluator.matmul(harr2, harr1), nparr2.T @ nparr1)

        nparr2 = np.random.randint(-10000, 10000, (64,))
        harr2 = self.kit.array(nparr2)
        self.assert_array_equal(self.evaluator.matmul(harr1, harr2), nparr1 @ nparr2)

    def test_evaluate_parallel(self):
        nparr1 = np.random.randint(-10000, 10000, (100, 100))
        harr1 = self.kit.array(nparr1)
        nparr2 = np.random.randint(-10000, 10000, (100, 100))
        harr2 = self.encryptor.encrypt(self.kit.array(nparr2))

        # to_bytes
        self.assertEqual(nparr1.dtype, np.int64)
        self.assertEqual(harr1.to_bytes(8, sys.byteorder), nparr1.tobytes())

        # pt - pt
        self.assert_array_equal(self.evaluator.add(harr1, harr1), nparr1 * 2)
        self.assert_array_equal(self.evaluator.sub(harr1, harr1), nparr1 - nparr1)
        self.assert_array_equal(
            self.evaluator.mul(harr1, harr1), np.multiply(nparr1, nparr1)
        )
        self.assert_array_equal(self.evaluator.matmul(harr1, harr1), nparr1 @ nparr1)
        self.assertEqual(
            self.evaluator.sum(harr1),
            phe.Plaintext(self.kit.get_schema(), int(nparr1.sum())),
        )

        # ct - pt
        self.assert_array_equal((self.evaluator.add(harr2, harr1)), nparr2 + nparr1)
        self.assert_array_equal((self.evaluator.sub(harr2, harr1)), nparr2 - nparr1)
        self.assert_array_equal(
            (self.evaluator.mul(harr2, harr1)), np.multiply(nparr2, nparr1)
        )
        self.assert_array_equal((self.evaluator.matmul(harr2, harr1)), nparr2 @ nparr1)
        self.assertEqual(
            self.decryptor.decrypt(self.evaluator.sum(harr2)),
            phe.Plaintext(self.kit.get_schema(), int(nparr2.sum())),
        )

        # pt - ct
        self.assert_array_equal(self.evaluator.add(harr1, harr2), nparr1 + nparr2)
        self.assert_array_equal(self.evaluator.sub(harr1, harr2), nparr1 - nparr2)
        self.assert_array_equal(
            self.evaluator.mul(harr1, harr2), np.multiply(nparr1, nparr2)
        )
        self.assert_array_equal(self.evaluator.matmul(harr1, harr2), nparr1 @ nparr2)

        # ct - ct
        self.assert_array_equal(self.evaluator.add(harr2, harr2), nparr2 + nparr2)
        self.assert_array_equal(self.evaluator.sub(harr2, harr2), nparr2 - nparr2)

    def test_serialize(self):
        edr_params = pickle.dumps(phe.BigintEncoderParams())

        # client: encrypt and send
        pk_buffer = pickle.dumps(self.kit.public_key())
        pt_buffer = pickle.dumps(
            self.kit.array([[16, 19], [36, 43]], pickle.loads(edr_params))
        )
        ct_buffer = pickle.dumps(self.encryptor.encrypt(self.kit.array([1, 2])))

        # server: calc ct1 - ct2
        server_he = hnp.setup(pickle.loads(pk_buffer))
        pt = pickle.loads(pt_buffer)
        ct = pickle.loads(ct_buffer)
        res = server_he.evaluator().matmul(pt, ct)
        res_buffer = pickle.dumps(res)

        # client: decrypt
        res = pickle.loads(res_buffer)
        self.assert_array_equal(
            self.decryptor.decrypt(res),
            np.array([[16, 19], [36, 43]]) @ np.array([1, 2]),
        )

    def test_serialize_ic(self):
        arr = self.kit.array([[16, 19], [36, 43]])
        buf1 = arr.serialize(hnp.MatrixSerializeFormat.Interconnection)
        buf2 = arr.serialize()
        self.assertNotEquals(buf1, buf2)
        arr2 = arr.load_from(buf1, hnp.MatrixSerializeFormat.Interconnection)
        self.assert_array_equal(arr2, arr.to_numpy())
        arr2 = arr.load_from(buf2)
        self.assert_array_equal(arr2, arr.to_numpy())

        arr = self.kit.array([100])
        buf = arr.serialize(hnp.MatrixSerializeFormat.Interconnection)
        arr2 = arr.load_from(buf, hnp.MatrixSerializeFormat.Interconnection)
        self.assert_array_equal(arr2, arr.to_numpy())

    def test_slice_get(self):
        nparr = np.arange(49).reshape((7, 7))
        harr = self.kit.array(nparr)

        self.assert_array_equal(harr[5:1, 1], nparr[5:1, 1])
        self.assert_array_equal(harr[4:, [1, 2, 2, 3]], nparr[4:, [1, 2, 2, 3]])
        self.assert_array_equal(harr[:4, (5,)], nparr[:4, (5,)])
        self.assert_array_equal(harr[-3:-1, (1, 3)], nparr[-3:-1, (1, 3)])
        self.assert_array_equal(harr[1:5:2, -1], nparr[1:5:2, -1])
        self.assert_array_equal(harr[::2, [-7, 5]], nparr[::2, [-7, 5]])
        self.assert_array_equal(harr[[1, 2, 3]], nparr[[1, 2, 3]])
        self.assert_array_equal(harr[1], nparr[1])
        self.assert_array_equal(harr[:], nparr[:])

        nparr = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        harr = self.kit.array(nparr)
        self.assertEqual(tuple(harr.shape), (8,))
        self.assert_array_equal(harr[[1, 2, 3]], nparr[[1, 2, 3]])
        self.assert_array_equal(harr[:], nparr[:])

        self.assertEqual(int(harr[1]), nparr[1])
        self.assertEqual(int(harr[-1]), nparr[-1])
        self.assertEqual(int(harr[-7]), nparr[-7])

        # warning for numpy users: below slices are totally different:
        # heu-tensor returns a 2x2 matrix, while numpy returns 2-len vector
        # self.assert_array_equal(harr[[1, 2], [0, -1]], nparr[[1, 2], [0, -1]])

        # warning for numpy users: below slices are different
        # harr[(0,)]  ->  got 1d array: [0]
        # nparr[(0,)] ->  get scalar: 0

    def test_slice_set(self):
        # 2d <- 2d case
        nparr = np.arange(49).reshape((7, 7))
        harr = self.kit.array(nparr)

        nppatch = np.arange(16).reshape((4, 4))
        hpatch = self.kit.array(nppatch)
        nparr[1:5, [0, 1, 5, 6]] = nppatch
        harr[1:5, [0, 1, 5, 6]] = hpatch
        self.assert_array_equal(harr, nparr)
        nparr[[5, 6], 6:4:-1] = nparr[0:2, [3, 2]]
        harr[[5, 6], 6:4:-1] = harr[0:2, [3, 2]]
        self.assert_array_equal(harr, nparr)

        # 2d <- 1d case
        nppatch = np.arange(7)
        hpatch = self.kit.array(nppatch)
        nparr[0, :] = nppatch
        harr[0, :] = hpatch
        self.assert_array_equal(harr, nparr)

        nparr[:, 5] = nppatch
        harr[:, 5] = hpatch
        self.assert_array_equal(harr, nparr)

        # 2d <- scalar case
        nparr[1, 1] = 100
        harr[1, 1] = phe.Plaintext(self.kit.get_schema(), 100)
        self.assert_array_equal(harr, nparr)

        # 1d <- 1d case
        nparr = nparr[6, :]
        harr = harr[6, :]
        nppatch = np.arange(2)
        hpatch = self.kit.array(nppatch)
        nparr[[2, 5]] = nppatch
        harr[[2, 5]] = hpatch
        self.assert_array_equal(harr, nparr)

        # 1d <- scalar case
        nparr[-1] = 666
        harr[-1] = phe.Plaintext(self.kit.get_schema(), 666)
        self.assert_array_equal(harr, nparr)

    def test_ciphertext_slice_2d(self):
        nparr = np.arange(49).reshape((7, 7))
        harr = self.kit.array(nparr)
        carr = self.encryptor.encrypt(harr)
        # scalar case
        carr[6, 6] = self.encryptor.phe.encrypt(
            phe.Plaintext(self.kit.get_schema(), 666)
        )
        carr[6, 6] = self.encryptor.phe.encrypt_raw(1000)
        nparr[6, 6] = 1000
        # block case
        carr[[5, 6], 6:4:-1] = carr[0:2, [3, 2]]
        nparr[[5, 6], 6:4:-1] = nparr[0:2, [3, 2]]
        harr = self.decryptor.decrypt(carr)
        self.assert_array_equal(harr, nparr)
        # row case
        carr[:, -1] = carr[0, :]
        nparr[:, -1] = nparr[0, :]
        harr = self.decryptor.decrypt(carr)
        self.assert_array_equal(harr, nparr)
        # col case
        carr[-2, :] = carr[0, :]
        nparr[-2, :] = nparr[0, :]
        harr = self.decryptor.decrypt(carr)
        self.assert_array_equal(harr, nparr)

    def test_ciphertext_slice_1d(self):
        nparr = np.arange(100)
        harr = self.kit.array(nparr)
        carr = self.encryptor.encrypt(harr)
        # scalar case
        carr[-100] = self.encryptor.phe.encrypt_raw(1000)
        nparr[-100] = 1000
        # 1d case
        carr[1:3] = carr[98:]
        nparr[1:3] = nparr[98:]
        harr = self.decryptor.decrypt(carr)
        self.assert_array_equal(harr, nparr)

    def test_shape_and_random(self):
        m1 = hnp.random.randint(
            phe.Plaintext(self.kit.get_schema(), -100),
            phe.Plaintext(self.kit.get_schema(), 100),
            (10,),
        )
        self.assertEqual(m1.ndim, 1)
        self.assertEqual(tuple(m1.shape), (10,))

        buf = pickle.dumps(hnp.Shape(10, 20, 30))
        sp = pickle.loads(buf)[:2]
        self.assertTrue(isinstance(sp, hnp.Shape))
        m1 = hnp.random.randbits(self.kit.get_schema(), 2048, sp)
        self.assertEqual(m1.ndim, 2)
        self.assertEqual(tuple(m1.shape), (10, 20))

    def test_batch_select_sum(self):
        sample_size = 100 * 10000
        m1 = hnp.random.randint(
            phe.Plaintext(self.kit.get_schema(), -100),
            phe.Plaintext(self.kit.get_schema(), 100),
            (sample_size, 2),
        )

        # select randon 150 row indices and create 50000 batches
        select_indices = [
            np.random.randint(0, sample_size, 150).tolist() for _ in range(50000)
        ]
        batch_select_result = self.evaluator.batch_select_sum(m1, select_indices)
        true_results = [self.evaluator.sum(m1[a]) for a in select_indices]
        for i in range(len(true_results)):
            self.assertEqual(batch_select_result[i], true_results[i])

    def test_feature_wise_bucket_sum(self):
        sample_size = 450
        feature_size = 10
        m1 = hnp.random.randint(
            phe.Plaintext(self.kit.get_schema(), -100),
            phe.Plaintext(self.kit.get_schema(), 100),
            (sample_size, 2),
        )
        first_group = np.zeros(sample_size, dtype=np.int8)
        second_group = np.zeros(sample_size, dtype=np.int8)
        first_group[:50] = 1
        second_group[50:] = 1
        subgroup_map = [first_group, second_group]

        order_map = np.zeros((sample_size, feature_size), dtype=np.int8)
        # first 50 samples goes to bucket 0, last 50 bucket 1.
        order_map[50:, :] = 1
        bucket_num = 2
        # sums should be a list of length 2
        # each element should be a matrix of size 4 * 2
        first_result = self.evaluator.feature_wise_bucket_sum(
            m1, subgroup_map[0], order_map, bucket_num, False
        )
        assert first_result[0, 0] == self.evaluator.select_sum(
            m1[:, 0], [i for i in range(50)]
        )
        assert first_result[0, 1] == self.evaluator.select_sum(
            m1[:, 1], [i for i in range(50)]
        )
        assert first_result[1, 0] == self.evaluator.select_sum(m1, [])
        assert first_result[1, 1] == self.evaluator.select_sum(m1, [])

    def test_batch_feature_wise_bucket_sum(self):
        sample_size = 100
        m1 = hnp.random.randint(
            phe.Plaintext(self.kit.get_schema(), -100),
            phe.Plaintext(self.kit.get_schema(), 100),
            (sample_size, 2),
        )
        # first 50 in group 1, last 50 in group 2
        first_group = np.zeros(sample_size, dtype=np.int8)
        second_group = np.zeros(sample_size, dtype=np.int8)
        first_group[:50] = 1
        second_group[50:] = 1
        subgroup_map = [first_group, second_group]

        # 2 feature,
        order_map = np.zeros((sample_size, 2), dtype=np.int8)
        # first 50 samples goes to bucket 0, last 50 bucket 1.
        order_map[50:, :] = 1
        bucket_num = 2
        # sums should be a list of length 2
        # each element should be a matrix of size 4 * 2
        sums = self.evaluator.batch_feature_wise_bucket_sum(
            m1, subgroup_map, order_map, bucket_num, False
        )
        first_result = sums[0]
        assert first_result[0, 0] == self.evaluator.select_sum(
            m1[:, 0], [i for i in range(50)]
        )
        assert first_result[0, 1] == self.evaluator.select_sum(
            m1[:, 1], [i for i in range(50)]
        )
        assert first_result[1, 0] == self.evaluator.select_sum(m1, [])
        assert first_result[1, 1] == self.evaluator.select_sum(m1, [])
        second_result = sums[1]
        assert second_result[0, 0] == self.evaluator.select_sum(m1, [])
        assert second_result[0, 1] == self.evaluator.select_sum(m1, [])
        assert second_result[1, 0] == self.evaluator.select_sum(
            m1[:, 0], [i for i in range(50, 100, 1)]
        )
        assert second_result[1, 1] == self.evaluator.select_sum(
            m1[:, 1], [i for i in range(50, 100, 1)]
        )

    def test_tree_predict(self):
        x = np.array(
            [
                [
                    -0.51422644,
                    0.73001039,
                    -0.7303912,
                    0.97048271,
                    -0.35085386,
                    -0.80080819,
                    -0.2015295,
                    -0.49920642,
                    -0.75011241,
                    -0.9106403,
                ],
                [
                    -0.72553682,
                    0.48224366,
                    -0.82322264,
                    0.20211923,
                    -0.27067894,
                    -0.13978112,
                    0.36809838,
                    -0.65290093,
                    0.43806458,
                    0.83020592,
                ],
            ]
        )
        split_features = [-1, -1, 3, -1, -1, -1, 0]
        split_points = [
            np.inf,
            np.inf,
            0.0699642896652222,
            np.inf,
            np.inf,
            np.inf,
            0.999987125396729,
        ]
        correct_selects = np.array([[1, 1, 1, 1, 0, 0, 1, 0], [1, 1, 1, 1, 0, 0, 1, 0]])
        assert (
            hnp.tree_predict(x, split_features, split_points) == correct_selects
        ).all()

    def test_tree_predict_with_indices(self):
        x = np.array(
            [
                [
                    -0.51422644,
                    0.73001039,
                    -0.7303912,
                    0.97048271,
                    -0.35085386,
                    -0.80080819,
                    -0.2015295,
                    -0.49920642,
                    -0.75011241,
                    -0.9106403,
                ],
                [
                    -0.72553682,
                    0.48224366,
                    -0.82322264,
                    0.20211923,
                    -0.27067894,
                    -0.13978112,
                    0.36809838,
                    -0.65290093,
                    0.43806458,
                    0.83020592,
                ],
            ]
        )
        split_features = [-1, -1, 3, -1, -1, -1, 0]
        num_split_node = len(split_features)
        leaf_num = num_split_node + 1
        split_points = [
            np.inf,
            np.inf,
            0.0699642896652222,
            np.inf,
            np.inf,
            np.inf,
            0.999987125396729,
        ]
        indices = [*range(num_split_node)]
        leaf_indices = [num_split_node + i for i in range(leaf_num)]
        correct_selects = np.array([[1, 1, 1, 1, 0, 0, 1, 0], [1, 1, 1, 1, 0, 0, 1, 0]])
        assert (
            hnp.tree_predict_with_indices(
                x, split_features, split_points, indices, leaf_indices
            )
            == correct_selects
        ).all()


if __name__ == "__main__":
    unittest.main()

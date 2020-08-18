import unittest
from reference.annotated_transformer_one_doc import *

# class TestAnnotatedTransformer(unittest.TestCase):
#
#     def test_subsequent_mask(self):
#         mask = subsequent_mask(5)
#         gold = np.array(
#             [[1,0,0,0,0],
#              [1,1,0,0,0],
#              [1,1,1,0,0],
#              [1,1,1,1,0],
#              [1,1,1,1,1]]
#         )
#         print(mask.numpy())
#         assert (mask.squeeze(-1).numpy() == gold).all()
#

if __name__ == '__main__':
    unittest.main()
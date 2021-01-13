#!/usr/bin/env python3
import os
import unittest

TEST_FILE = Path(__file__).parents[0] / "testfiles" / "dummy.tif"


class MaskTest(unittest.TestCase):
    def setUp(self):
        self.img = Image(TEST_FILE)

    def tearDown(self):
        self.img.close()

    

if __name__ == "__main__":
    unittest.main()

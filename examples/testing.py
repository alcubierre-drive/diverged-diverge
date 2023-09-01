#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import diverge
diverge.mpi_init(None, None)
diverge.run_tests()
diverge.mpi_finalize()

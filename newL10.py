L10 = np.array([
                # C1
                [# M = 0.2
                 [# N = 0.003
                  [0.0, 0.1, 0.1, 1.4, 2.1, 3.4],
                  # N = 0.006
                  [0.2, 0.1, 0.1, 0.0, 1.4, 2.7],
                  # N = 0.024
                  [0.2, 0.1, 0.1, 0.0, 1.4, 2.7]],

                 # M = 0.4
                 [# N = 0.003
                  [0.0, 0.1, 0.1, 0.5, 2.4, 3.4],
                  # N = 0.006
                  [0.2, 0.1, -0.3, 0.3, 1.1, 2.4],
                  # N = 0.024
                  [0.2, 0.1, -0.3, 0.3, 1.1, 2.4]],

                 # M = 0.5
                 [# N = 0.003
                  [0.1, 0.1, 0.1, 0.5, 2.4, 3.0],
                  # N = 0.006
                  [0.4, 0.1, -0.1, 0.3, 1.5, 2.2],
                  # N = 0.024
                  [0.4, 0.1, -0.1, 0.3, 1.5, 2.2]],

                 # M = 0.6
                 [# N = 0.003
                  [0.1, 0.2, 0.1, 0.5, 1.9, 3.2],
                  # N = 0.006
                  [0.4, 0.2, 0.0, 0.3, 1.6, 1.9],
                  # N = 0.024
                  [0.4, 0.2, 0.0, 0.3, 1.6, 1.9]],

                 # M = 0.8
                 [# N = 0.003
                  [0.1, 0.4, 0.1, 1.1, 2.2, 2.4],
                  # N = 0.006
                  [0.7, 0.3, 0.1, 0.4, 1.9, 1.9],
                  # N = 0.024
                  [0.7, 0.3, 0.1, 0.4, 1.9, 1.9]],

                 # M = 1.0
                 [# N = 0.003
                  [0.4, 0.4, 0.1, 1.3, 2.2, 2.3],
                  # N = 0.006
                  [0.7, 0.4, 0.1, 0.4, 2.2, 1.5],
                  # N = 0.024
                  [0.7, 0.4, 0.1, 0.4, 2.2, 1.5]],

                 # M = 1.5
                 [# N = 0.003
                  [0.4, 0.4, 0.1, 1.2, 2.4, 2.3],
                  # N = 0.006
                  [1.1, 0.8, 0.5, 0.8, 2.0, 1.5],
                  # N = 0.024
                  [1.1, 0.8, 0.5, 0.8, 2.0, 1.5]],

                 # M = 2.0
                 [# N = 0.003
                  [0.7, 0.7, 0.1, 1.2, 2.4, 2.3],
                  # N = 0.006
                  [1.6, 1.5, 0.9, 0.8, 2.0, 1.5],
                  # N = 0.024
                  [1.6, 1.5, 0.9, 0.8, 2.0, 1.5]],

                 # M = 4.0
                 [# N = 0.003
                  [1.4, 1.4, 0.1, 1.2, 2.4, 2.3],
                  # N = 0.006
                  [1.9, 1.5, 1.7, 0.8, 2.0, 1.5],
                  # N = 0.024
                  [1.9, 1.5, 1.7, 0.8, 2.0, 1.5]],

                 # M = 10.0
                 [# N = 0.003
                  [1.4, 1.4, 0.1, 1.4, 2.4, 2.3],
                  # N = 0.006
                  [1.9, 1.5, 1.7, 0.8, 2.0, 1.5],
                  # N = 0.024
                  [1.9, 1.5, 1.7, 0.8, 2.0, 1.5]]],

                # C2
                [# M = 0.2
                 [# N = 0.003
                  [9.0, 9.0, 8.0, 3.0, 1.0, 1.0],
                  # N = 0.006
                  [5.0, 7.0, 8.0, 8.0, 4.0, 3.0],
                  # N = 0.024
                  [5.0, 7.0, 8.0, 8.0, 4.0, 3.0]],

                 # M = 0.4
                 [# N = 0.003
                  [8.0, 8.0, 7.0, 6.0, 2.0, 1.0],
                  # N = 0.006
                  [5.0, 7.0, 8.0, 6.0, 7.0, 4.0],
                  # N = 0.024
                  [5.0, 7.0, 8.0, 6.0, 7.0, 4.0]],

                 # M = 0.5
                 [# N = 0.003
                  [7.0, 8.0, 7.0, 6.0, 2.0, 4.0],
                  # N = 0.006
                  [4.0, 7.0, 7.0, 6.0, 5.0, 7.0],
                  # N = 0.024
                  [4.0, 7.0, 7.0, 6.0, 5.0, 7.0]],

                 # M = 0.6
                 [# N = 0.003
                  [7.0, 8.0, 7.0, 6.0, 6.0, 6.0],
                  # N = 0.006
                  [4.0, 6.0, 7.0, 6.0, 5.0, 9.0],
                  # N = 0.024
                  [4.0, 6.0, 7.0, 6.0, 5.0, 9.0]],

                 # M = 0.8
                 [# N = 0.003
                  [7.0, 7.0, 7.0, 4.0, 8.0, 12.0],
                  # N = 0.006
                  [3.0, 6.0, 6.0, 7.0, 6.0, 12.0],
                  # N = 0.024
                  [3.0, 6.0, 6.0, 7.0, 6.0, 12.0]],

                 # M = 1.0
                 [# N = 0.003
                  [5.0, 7.0, 7.0, 4.0, 8.0, 12.0],
                  # N = 0.006
                  [3.0, 6.0, 6.0, 7.0, 6.0, 14.0],
                  # N = 0.024
                  [3.0, 6.0, 6.0, 7.0, 6.0, 14.0]],

                 # M = 1.5
                 [# N = 0.003
                  [5.0, 5.0, 7.0, 7.0, 8.0, 12.0],
                  # N = 0.006
                  [2.0, 4.0, 3.0, 6.0, 8.0, 14.0],
                  # N = 0.024
                  [2.0, 4.0, 3.0, 6.0, 8.0, 14.0]],

                 # M = 2.0
                 [# N = 0.003
                  [3.0, 3.0, 7.0, 7.0, 8.0, 12.0],
                  # N = 0.006
                  [0.0, 0.0, 2.0, 6.0, 8.0, 14.0],
                  # N = 0.024
                  [0.0, 0.0, 2.0, 6.0, 8.0, 14.0]],

                 # M = 4.0
                 [# N = 0.003
                  [1.0, 1.0, 7.0, 7.0, 8.0, 12.0],
                  # N = 0.006
                  [0.0, 0.0, 0.0, 6.0, 8.0, 14.0],
                  # N = 0.024
                  [0.0, 0.0, 0.0, 6.0, 8.0, 14.0]],

                 # M = 10.0
                 [# N = 0.003
                  [1.0, 1.0, 7.0, 6.0, 8.0, 12.0],
                  # N = 0.006
                  [0.0, 0.0, 0.0, 6.0, 8.0, 14.0],
                  # N = 0.024
                  [0.0, 0.0, 0.0, 6.0, 8.0, 14.0]]],

                # C3
                [# M = 0.2
                 [# N = 0.003
                  [0.6, 0.6, 0.5, 0.5, 0.5, 0.5],
                  # N = 0.006
                  [0.5, 0.6, 0.5, 0.5, 0.5, 0.5],
                  # N = 0.024
                  [0.5, 0.6, 0.5, 0.5, 0.5, 0.5]],

                 # M = 0.4
                 [# N = 0.003
                  [0.6, 0.6, 0.5, 0.5, 0.5, 0.5],
                  # N = 0.006
                  [0.5, 0.6, 0.5, 0.5, 0.5, 0.5],
                  # N = 0.024
                  [0.5, 0.6, 0.5, 0.5, 0.5, 0.5]],

                 # M = 0.5
                 [# N = 0.003
                  [0.6, 0.6, 0.5, 0.5, 0.5, 0.5],
                  # N = 0.006
                  [0.5, 0.6, 0.5, 0.5, 0.5, 0.5],
                  # N = 0.024
                  [0.5, 0.6, 0.5, 0.5, 0.5, 0.5]],

                 # M = 0.6
                 [# N = 0.003
                  [0.6, 0.6, 0.5, 0.5, 0.5, 0.5],
                  # N = 0.006
                  [0.5, 0.6, 0.5, 0.5, 0.5, 0.5],
                  # N = 0.024
                  [0.5, 0.6, 0.5, 0.5, 0.5, 0.5]],

                 # M = 0.8
                 [# N = 0.003
                  [0.6, 0.6, 0.5, 0.5, 0.5, 0.5],
                  # N = 0.006
                  [0.5, 0.6, 0.5, 0.5, 0.5, 0.5],
                  # N = 0.024
                  [0.5, 0.6, 0.5, 0.5, 0.5, 0.5]],

                 # M = 1.0
                 [# N = 0.003
                  [0.6, 0.6, 0.5, 0.5, 0.5, 0.5],
                  # N = 0.006
                  [0.5, 0.6, 0.5, 0.5, 0.5, 0.5],
                  # N = 0.024
                  [0.5, 0.6, 0.5, 0.5, 0.5, 0.5]],

                 # M = 1.5
                 [# N = 0.003
                  [0.6, 0.6, 0.5, 0.5, 0.5, 0.5],
                  # N = 0.006
                  [0.5, 0.6, 0.5, 0.5, 0.5, 0.5],
                  # N = 0.024
                  [0.5, 0.6, 0.5, 0.5, 0.5, 0.5]],

                 # M = 2.0
                 [# N = 0.003
                  [0.6, 0.6, 0.5, 0.5, 0.5, 0.5],
                  # N = 0.006
                  [0.5, 0.6, 0.5, 0.5, 0.5, 0.5],
                  # N = 0.024
                  [0.5, 0.6, 0.5, 0.5, 0.5, 0.5]],

                 # M = 4.0
                 [# N = 0.003
                  [0.6, 0.6, 0.5, 0.5, 0.5, 0.5],
                  # N = 0.006
                  [0.5, 0.6, 0.5, 0.5, 0.5, 0.5],
                  # N = 0.024
                  [0.5, 0.6, 0.5, 0.5, 0.5, 0.5]],

                 # M = 10.0
                 [# N = 0.003
                  [0.6, 0.6, 0.5, 0.5, 0.5, 0.5],
                  # N = 0.006
                  [0.5, 0.6, 0.5, 0.5, 0.5, 0.5],
                  # N = 0.024
                  [0.5, 0.6, 0.5, 0.5, 0.5, 0.5]]],

                # Kb
                [# M = 0.2
                 [# N = 0.003
                  [10.0, 10.0, 15.0, 15.0, 15.0, 20.0],
                  # N = 0.006
                  [15.0, 10.0, 10.0, 10.0, 15.0, 20.0],
                  # N = 0.024
                  [15.0, 10.0, 10.0, 10.0, 15.0, 20.0]],

                 # M = 0.4
                 [# N = 0.003
                  [10.0, 10.0, 10.0, 5.0, 15.0, 20.0],
                  # N = 0.006
                  [15.0, 10.0, 10.0, 5.0, 15.0, 20.0],
                  # N = 0.024
                  [15.0, 10.0, 10.0, 5.0, 15.0, 20.0]],

                 # M = 0.5
                 [# N = 0.003
                  [10.0, 10.0, 10.0, 5.0, 15.0, 20.0],
                  # N = 0.006
                  [15.0, 10.0, 10.0, 5.0, 15.0, 20.0],
                  # N = 0.024
                  [15.0, 10.0, 10.0, 5.0, 15.0, 20.0]],

                 # M = 0.6
                 [# N = 0.003
                  [10.0, 10.0, 10.0, 5.0, 15.0, 15.0],
                  # N = 0.006
                  [15.0, 10.0, 10.0, 5.0, 15.0, 15.0],
                  # N = 0.024
                  [15.0, 10.0, 10.0, 5.0, 15.0, 15.0]],

                 # M = 0.8
                 [# N = 0.003
                  [10.0, 5.0, 5.0, 5.0, 15.0, 15.0],
                  # N = 0.006
                  [15.0, 10.0, 5.0, 5.0, 15.0, 15.0],
                  # N = 0.024
                  [15.0, 10.0, 5.0, 5.0, 15.0, 15.0]],

                 # M = 1.0
                 [# N = 0.003
                  [5.0, 5.0, 5.0, 10.0, 15.0, 15.0],
                  # N = 0.006
                  [15.0, 10.0, 5.0, 5.0, 15.0, 15.0],
                  # N = 0.024
                  [15.0, 10.0, 5.0, 5.0, 15.0, 15.0]],

                 # M = 1.5
                 [# N = 0.003
                  [5.0, 5.0, 5.0, 10.0, 15.0, 15.0],
                  # N = 0.006
                  [15.0, 10.0, 5.0, 10.0, 15.0, 15.0],
                  # N = 0.024
                  [15.0, 10.0, 5.0, 10.0, 15.0, 15.0]],

                 # M = 2.0
                 [# N = 0.003
                  [5.0, 5.0, 5.0, 10.0, 15.0, 15.0],
                  # N = 0.006
                  [15.0, 10.0, 5.0, 10.0, 15.0, 15.0],
                  # N = 0.024
                  [15.0, 10.0, 5.0, 10.0, 15.0, 15.0]],

                 # M = 4.0
                 [# N = 0.003
                  [5.0, 5.0, 5.0, 10.0, 15.0, 15.0],
                  # N = 0.006
                  [15.0, 10.0, 5.0, 10.0, 15.0, 15.0],
                  # N = 0.024
                  [15.0, 10.0, 5.0, 10.0, 15.0, 15.0]],

                 # M = 10.0
                 [# N = 0.003
                  [5.0, 5.0, 5.0, 10.0, 15.0, 15.0],
                  # N = 0.006
                  [15.0, 10.0, 5.0, 10.0, 15.0, 15.0],
                  # N = 0.024
                  [15.0, 10.0, 5.0, 10.0, 15.0, 15.0]]]])


import os
import sys
import subprocess


def fine_tune_scheme(op):
    tmp = {
        '1': [
            "sh -x ./run_ppl.sh usr_2step_fine_1M",
            "sh -x ./run_ppl.sh usr_1M_1M",
            "sh -x ./run_ppl.sh usr_2step_fine_500K",
            "sh -x ./run_ppl.sh usr_1M_500K",
            "sh -x ./run_ppl.sh usr_2step_fine_250K",
            "sh -x ./run_ppl.sh usr_1M_250K",

            "sh -x ./run_ppl.sh usr_2step_fine_200K",
            "sh -x ./run_ppl.sh usr_1M_200K",
            "sh -x ./run_ppl.sh usr_2step_fine_150K",
            "sh -x ./run_ppl.sh usr_1M_150K",

            "sh -x ./run_ppl.sh usr_2step_fine_100K",
            "sh -x ./run_ppl.sh usr_1M_100K",
            "sh -x ./run_ppl.sh usr_2step_fine_50K",
            "sh -x ./run_ppl.sh usr_1M_50K",

        ],

        '2':[
            "sh -x ./run_2step_ft_250K_100K.sh 200000 200K",
            "sh -x ./run_2step_ft_250K_100K.sh 150000 150K",
        ],

        '3':[
            "sh -x ./run_eval.sh",
            "sh -x ./run_eval_tp.sh",
        ],

        '4':[
            "sh -x ./run_ppl.sh",
            "sh -x ./run_eval.sh"
        ]

           }

    for cmd in tmp[op]:
        print('-'*20)
        print(cmd)
        print('-'*20)
        p = subprocess.Popen(cmd, shell=True)
        p.wait()
        print('\n'*2)


if __name__ == '__main__':
    fine_tune_scheme(op=sys.argv[1])



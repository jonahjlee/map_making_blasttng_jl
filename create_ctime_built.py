import os

import timing


def main():
    directory_path = "/media/player1/blast2020fc1/fc1/converted"

    for dir_name in os.listdir(directory_path):
        dir_path = os.path.join(directory_path, dir_name)

        if os.path.isdir(dir_path):
            print(f"dir: {dir_path}")

            if dir_name.startswith('master'):
                print(f"is master")
                t = timing.timing(master_path=dir_path)
                t.ctime_master(write=True)

            elif dir_name.startswith('roach'):
                print(f"is roach", end="")
                roach_number = dir_name[5] # 6th character
                print(f" number {roach_number}")
                t = timing.timing(roach_path=dir_path)
                print(f"created timing object")
                t.ctime_roach(roach_number=roach_number, kind=['Packet','Clock'],
                              mode='average', write=True)
                print(f"timing file written")


if __name__ == '__main__':
     main()
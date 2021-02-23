import os
import io

def _remove_file(file):
    if os.path.isfile(file):
        os.remove(file)
        return True
    return False

def split_file(file_path, src_file, trg_file, max_lines=None, lower=True):

    _remove_file(src_file)
    _remove_file(trg_file)

    with io.open(os.path.expanduser(file_path), encoding="utf8") as f:
        src_only_file = open(src_file, 'w')
        trg_only_file = open(trg_file, 'w')

        for i, line in enumerate(f):
            if lower:
                line = line.lower()
            line = line.strip()
            line = line.split("\t")

            if len(line) != 2 or line[0] == "" or line[1] == "":
                print(f"[WARN] bad line({i}):", line)
                continue

            if max_lines is not None and max_lines <= i:
                break

            src_line: str
            trg_line: str
            src_line = line[0]
            trg_line = line[1]

            src_only_file.write(src_line + "\n")
            trg_only_file.write(trg_line + "\n")

        src_only_file.close()
        trg_only_file.close()

    return

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--src-out-file", type=str, required=True)
    parser.add_argument("--trg-out-file", type=str, required=True)

    parser.add_argument("--max-lines", type=int)
    parser.add_argument("--no-lower", default=False, action='store_true')


    args = parser.parse_args()

    split_file(args.file, args.src_out_file, args.trg_out_file, max_lines=args.max_lines, lower=not args.no_lower)
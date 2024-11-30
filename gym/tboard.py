from tensorboard import program

tracking_address = "./ppo_addition_logs/" # the path of your log file.

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.main()
    print(f"Tensorflow listening on {url}")

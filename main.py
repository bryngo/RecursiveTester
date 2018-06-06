from environment import Environment
from environment_tests import test_env


def main():
    print("Commencing magic...")

    env = Environment()
    test_env(env)

if __name__ == "__main__":
    main()
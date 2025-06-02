import argparse
import torch
import gymnasium as gym
import minigrid
import sys
import os

from model import MiniGridACModel
from trainer import A3CTrainer


def parse_arguments():

    parser = argparse.ArgumentParser(description='A3C Training für MiniGrid')

    parser.add_argument('--env', type=str, default='MiniGrid-Empty-5x5-v0',
                       help='MiniGrid Environment Name (default: MiniGrid-Empty-5x5-v0)')
    

    
    parser.add_argument('--workers', type=int, default=4,
                       help='Anzahl paralleler Workers (default: 4)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning Rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount Factor (default: 0.99)')
    parser.add_argument('--n-steps', type=int, default=5,
                       help='N-step returns (default: 5)')
    parser.add_argument('--max-episodes', type=int, default=1000,
                       help='Max Episodes pro Worker (default: 1000)')
    


    parser.add_argument('--save-interval', type=int, default=10,
                       help='Speicher-Intervall in Minuten (default: 10)')
    parser.add_argument('--load-model', type=str, default=None,
                       help='Pfad zu einem gespeicherten Model zum Laden')
    



    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cpu, cuda, oder auto (default: auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random Seed für Reproduzierbarkeit (default: 42)')
    
    return parser.parse_args()


def setup_device(device_arg):

    if device_arg == 'auto':

        if torch.cuda.is_available():
            print("CUDA vefügbar, aber für A3C verwenden wir CPU (bessere Multiprocessing Performance)")
            device = torch.device('cpu')
        else:
            device = torch.device('cpu')
    
    elif device_arg == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"GPU Training aktiviert: {torch.cuda.get_device_name()}")
        else:
            print("CUDA nicht verfügbar, verwende CPU")
            device = torch.device('cpu')
    else:
        device = torch.device(device_arg)

    print(f"Device: {device}")

    return device


def validate_environment(env_name):

    try:
        print(f"Teste Environment: {env_name}")
        env = gym.make(env_name)

        obs_space = env.observation_space
        action_space = env.action_space

        print(f"Observation Space: {obs_space}")
        print(f"Action Space: {action_space} ({action_space.n} Aktionen)")

        obs, info = env.reset()
        print(f"Image Shape: {obs['image'].shape}")
        print(f"Direction: {obs['direction']}")

        env.close()
        return obs_space, action_space
    
    except Exception as e:
        print(f"Environment Error: {e}")
        print(f"Verfügbare MiniGrid Environments:")
        print("- MiniGrid-Empty-5x5-v0 (einfach)")
        print("- MiniGrid-DoorKey-5x5-v0 (mittelschwer)")
        print("- MiniGrid-MultiRoom-N2-S4-v0 (schwer)")
        sys.exit(1)



def print_training_info(args, obs_space, action_space):

    print("\n" + "="*60)
    print("A3C TRAINING SETUP")
    print("="*60)
    print(f"Environment: {args.env}")
    print(f"Workers: {args.workers}")
    print(f"Max Episodes pro Worker: {args.max_episodes}")
    print(f"Total Max Episodes: {args.workers * args.max_episodes}")
    print(f"Learning Rate: {args.lr}")
    print(f"Discount Factor (Gamma): {args.gamma}")
    print(f"N-Step Returns: {args.n_steps}")
    print(f"Save Interval: {args.save_interval} Minuten")
    print(f"Random Seed: {args.seed}")
    
    if args.load_model:
        print(f"Lade Model: {args.load_model}")
    
    print("\n MODEL ARCHITECTURE:")
    print(f"Input Shape: {obs_space['image'].shape}")
    print(f"Actions: {action_space.n}")
    print(f"Model: MiniGridACModel (Actor-Critic)")
    print("="*60 + "\n")


def main():

    print("A3C für Minigrid - Let's Go!")


    args = parse_arguments()

    torch.manual_seed(args.seed)
    torch.random.manual_seed(args.seed)

    device = setup_device(args.device)

    obs_space, action_space = validate_environment(args.env)

    print_training_info(args, obs_space, action_space)


    print("Erstelle Training Manager...")
    training_manager = A3CTrainer(
        model_class=MiniGridACModel,
        obs_space=obs_space,
        action_space=action_space,
        env_name=args.env,
        num_workers=args.workers,
        lr=args.lr,
        gamma=args.gamma,
        n_steps=args.n_steps,
        max_episodes_per_worker=args.max_episodes
    )

    if args.load_model:
        if os.path.exists(args.load_model):
            training_manager.load_model(args.load_model)
        else:
            print(f" Model nicht gefunden: {args.load_model}")
            sys.exit(1)
    
    
    print("\nBereit zum Training!")
    print("Tipps:")
    print("- Drücke Ctrl+C um das Training zu stoppen")
    print("- Models werden automatisch gespeichert")
    print("- Statistiken werden alle 50 Episodes angezeigt")
    
    try:
        input("\n Drücke Enter um zu starten, oder Ctrl+C zum Abbrechen...")
    except KeyboardInterrupt:
        print("\n Training abgebrochen!")
        sys.exit(0)


    try:
        print(f"\n Training startet mit {args.workers} Workers!")
        training_manager.train(save_interval_minutes=args.save_interval)
        
    except KeyboardInterrupt:
        print(f"\n Training durch Benutzer gestoppt!")
        
    except Exception as e:
        print(f"\n Training Fehler: {e}")
        raise
    
    finally:
        print("\n Training Script beendet!")


if __name__ == "__main__":
        main()
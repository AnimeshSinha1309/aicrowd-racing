class ConfigurationSAC:

    VAE_LATENT_DIMS = 32
    VISION_HIDDEN_LAYER_SIZES = [32, 64, 64, 32, 32]
    SPEED_HIDDEN_LAYER_SIZES = [8, 8]
    ACTOR_HIDDEN_LAYER_SIZES = [64, 64, 32]
    REPLAY_BATCH_SIZE = 256
    REPLAY_MAX_STORAGE = 250_000
    START_STEPS = 2000
    UPDATE_AFTER = 2000
    UPDATE_EVERY_STEPS = 1
    EVAL_EVERY_STEPS = 5000
    MAX_EPISODE_LENGTH = 50000

    GAMMA = 0.99
    POLYAK = 0.995
    ALPHA = 0.2
    LEARNING_RATE = 0.003

    make_random_actions = 0
    load_checkpoint = False
    record_experience = False
    num_test_episodes = 1
    safety_margin = 4.2
    save_episodes = 1
    save_freq = 1
    total_steps = 250_000
    start_steps = 2000

    checkpoint = "models/sac/checkpoints/best_sac_local_encoder-vae_small_seed-249_episode_480.statedict"
    model_save_path = "saved/l2r/results/${DIRHASH}workspaces/${USER}/results"
    track_name = "Thruxton"
    experiment_name = "SoftActorCritic"
    safety_data = "saved/l2r/datasets/l2r/datasets/safety_sets"
    record_dir = "saved/l2r/datasets/l2r/datasets/safety_records_dataset/"

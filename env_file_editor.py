import yaml

def remove_build_info(env_yaml_path):
    """
    Remove the build information from the dependiencies in the environment.yaml file

    :param env_yaml_path: (str) location of environment.yaml file
    """
    with open(env_yaml_path, 'r') as file:
        env_yaml = yaml.safe_load(file)

    # Modify dependencies to remove build information
    if 'dependencies' in env_yaml:
        for i, dep in enumerate(env_yaml['dependencies']):
            parts = dep.split('=')
            env_yaml['dependencies'][i] = parts[0] if len(parts) == 1 else '='.join(parts[:-1])

    with open(env_yaml_path, 'w') as file:
        yaml.dump(env_yaml, file)

if __name__ == "__main__":
    env_yaml_path = 'environment.yaml'  # Replace with the actual path
    remove_build_info(env_yaml_path)

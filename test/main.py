

if __name__ == "__main__":
    logging.basicConfig(filename='logs/RobotTest.log', level=logging.INFO)

    segmentation_model = Seg_detector.Segment()
    robot = robot_env.Robot(socket_ip1, socket_ip2, segmentation_model, args.seg_threshold)

    agent = Agent(robot)
    agent.run()
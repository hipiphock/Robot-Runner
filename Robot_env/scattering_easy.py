"""
scattering을 하기 위해서는 무엇을 해야할까?

Scenario
1. 물건 두 개가 겹쳐 있다.
2. 물건 두 개가 옆에 나란히 있다.

Solution for no.1
1. 가장 위에 있는 object를 집는다.
2. 아무 물건이 없는 곳에 치운다.

Solution for no.2
1. 두 물건 사이에 있는 틈으로 gripper를 close한 채로 밀어넣는다.
2. 옆에 있는 물건을 옆으로 친다, 또는 밀어낸다.

Logistics
1. Get neightboring objects from one object
    1. select one object.
    2. get distance from other objects
    3. if distance is small, then that is neighboring object
2. 
"""

def get_neighbor_obj():
    # 어떻게 detect 해야할까?
    # object의 경계선은 어디에 있지?
    return [1, 2, 3]

def put_neighbor_obj(obj):
    # neighboring object 옮기기
    # 어떻게 옮기지?
    pass

def scatter():
    neighbor_obj_list = get_neighbor_obj()
    for neighbor_obj in neighbor_obj_list:
        put_neighbor_obj(neighbor_obj)
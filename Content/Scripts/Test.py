import unreal_engine as ue
from unreal_engine.classes import GameplayStatics

class Test:
    def begin_play(self):
        player = GameplayStatics.GetPlayerController(self.uobject)
        self.uobject.EnableInput(player)
        self.uobject.bind_key('C', ue.IE_PRESSED, self.you_pressed_C)

    def you_pressed_C(self):
        print('you pressed C')
    
    def move_forward(self, amount):
        ue.print_string('axis value: ' + str(amount))

    def tick(self, delta_time):
        a = 1

 
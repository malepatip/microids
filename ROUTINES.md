# My Home Routines

Define your routines here. The LLM reads this file to match goals to device actions.
Use your actual device IDs from `microids devices list`.

## Leaving for work
When the user says "leaving for work" or "heading out", execute ALL of these tasks:
1. Open the garage door (device_id: cover.garage_door, capability: open)
2. Start the vacuum (device_id: vacuum.robot_vacuum, capability: start)
3. Water the front lawn (device_id: switch.front_sprinkler, capability: turn_on)
All three are independent — no dependencies between them.

## Coming home
When the user says "coming home" or "I'm home", execute ALL of these tasks:
1. Open the garage door (device_id: cover.garage_door, capability: open)
2. Dock the vacuum (device_id: vacuum.robot_vacuum, capability: return_to_base)
3. Turn off sprinklers (device_id: switch.front_sprinkler, capability: turn_off)

## Goodnight
When the user says "goodnight" or "going to bed", execute ALL of these tasks:
1. Close the garage door (device_id: cover.garage_door, capability: close)
2. Turn off all lights (device_id: light.living_room, capability: turn_off)
3. Dock the vacuum (device_id: vacuum.robot_vacuum, capability: return_to_base)

IMPORTANT: Always execute ALL tasks listed in a routine. Never skip tasks.


# Dictionary of Lottery Structures

lotteries = {
        'lottery_1': {
            'name': 'Apple',
            'outcome_number': 4,
            'stake': 'lo',
            'max_payoff': 40,
            'min_payoff': -12,
            'description': 'Example lottery with four final outcomes',
            'periods': {
                0: [{'label': 'Start', 'probability': 1, 'from': None, 'abs_prob' : 1}],
                1: [
                    {'label': '+£10', 'probability': 0.6, 'from': 'Start', 'abs_prob' : 0.6},
                    {'label': '-£10', 'probability': 0.4, 'from': 'Start', 'abs_prob' : 0.4}
                ],
                2: [
                    {'label': '+£7', 'probability': 1, 'from': '+£10', 'abs_prob' : 0.6},
                    {'label': '-£12', 'probability': 1, 'from': '-£10', 'abs_prob' : 0.4}
                ],
                3: [
                    {'label': '+£8', 'probability': 0.8, 'from': '+£7', 'parent': '+£10', 'abs_prob' : 0.48},
                    {'label': '+£0', 'probability': 0.2, 'from': '+£7', 'parent': '+£10', 'abs_prob' : 0.12},
                    {'label': '+£2', 'probability': 0.5, 'from': '-£12', 'parent': '-£10', 'abs_prob' : 0.2},
                    {'label': '+£5', 'probability': 0.5, 'from': '-£12', 'parent': '-£10', 'abs_prob' : 0.2},
                ]
            }
        },


        'lottery_2': {
            'name': 'Banana',
            'outcome_number': 6,
            'stake': 'hi',
            'max_payoff': 825,
            'min_payoff': -1245,
            'description': 'Example lottery with four final outcomes',
            'periods': {
                0: [{'label': 'Start', 'probability': 1, 'from': None, 'abs_prob' : 1}],
                1: [
                    {'label': '+£610', 'probability': 0.7, 'from': 'Start', 'abs_prob' : 0.6},
                    {'label': '+£645', 'probability': 0.3, 'from': 'Start', 'abs_prob' : 0.4}
                ],
                2: [
                    {'label': '-£665', 'probability': 1, 'from': '+£610', 'abs_prob' : 0.6},
                    {'label': '-£895', 'probability': 0.6, 'from': '+£645', 'abs_prob' : 0.4},
                    {'label': '-£800', 'probability': 0.4, 'from': '+£645', 'abs_prob' : 0.4}
                ],
                3: [
                    {'label': '+£865', 'probability': 0.3, 'from': '-£665', 'parent': '+£610', 'abs_prob' : 0.48},
                    {'label': '-£925', 'probability': 0.7, 'from': '-£665', 'parent': '+£610', 'abs_prob' : 0.12},
                    {'label': '+£940', 'probability': 0.6, 'from': '-£895', 'parent': '+£645', 'abs_prob' : 0.2},
                    {'label': '-£995', 'probability': 0.4, 'from': '-£895', 'parent': '+£645', 'abs_prob' : 0.2},
                    {'label': '-£860', 'probability': 0.6, 'from': '-£800', 'parent': '+£645', 'abs_prob' : 0.2},
                    {'label': '+£980', 'probability': 0.4, 'from': '-£800', 'parent': '+£645', 'abs_prob' : 0.2}
                ]
            }
        },
 
        'lottery_3': {
            'name': 'Lychee',
            'outcome_number': 4,
            'stake': 'hi',
            'max_payoff': 335,
            'min_payoff': -725,
            'description': 'Example lottery with four final outcomes',
            'periods': {
                0: [{'label': 'Start', 'probability': 1, 'from': None, 'abs_prob' : 1}],
                1: [
                    {'label': '+£120', 'probability': 0.8, 'from': 'Start', 'abs_prob' : 0.8},
                    {'label': '-£250', 'probability': 0.2, 'from': 'Start', 'abs_prob' : 0.2}
                ],
                2: [
                    {'label': '-£115', 'probability': 1, 'from': '+£120', 'abs_prob' : 0.8},
                    {'label': '+£120', 'probability': 1, 'from': '-£250', 'abs_prob' : 0.2}
                ],
                3: [
                    {'label': '+£210', 'probability': 0.8, 'from': '-£115', 'parent': '+£120', 'abs_prob' : 0.64},
                    {'label': '-£625', 'probability': 0.2, 'from': '-£115', 'parent': '+£120', 'abs_prob' : 0.16},
                    {'label': '-£595', 'probability': 0.5, 'from': '+£120', 'parent': '-£250', 'abs_prob' : 0.1},
                    {'label': '+£465', 'probability': 0.5, 'from': '+£120', 'parent': '-£250', 'abs_prob' : 0.1},
                ]
            }
        }
    }




one = {
        'lottery_1': {
            'name': 'Apple',
            'outcome_number': 4,
            'stake': 'lo',
            'max_payoff': 40,
            'min_payoff': -12,
            'description': 'Example lottery with four final outcomes',
            'periods': {
                0: [{'label': 'Start', 'probability': 1, 'from': None, 'abs_prob' : 1}],
                1: [
                    {'label': '+£10', 'probability': 0.6, 'from': 'Start', 'abs_prob' : 0.6},
                    {'label': '-£10', 'probability': 0.4, 'from': 'Start', 'abs_prob' : 0.4}
                ],
                2: [
                    {'label': '+£7', 'probability': 1, 'from': '+£10', 'abs_prob' : 0.6},
                    {'label': '-£12', 'probability': 1, 'from': '-£10', 'abs_prob' : 0.4}
                ],
                3: [
                    {'label': '+£8', 'probability': 0.8, 'from': '+£7', 'parent': '+£10', 'abs_prob' : 0.48},
                    {'label': '+£0', 'probability': 0.2, 'from': '+£7', 'parent': '+£10', 'abs_prob' : 0.12},
                    {'label': '+£2', 'probability': 0.5, 'from': '-£12', 'parent': '-£10', 'abs_prob' : 0.2},
                    {'label': '+£5', 'probability': 0.5, 'from': '-£12', 'parent': '-£10', 'abs_prob' : 0.2},
                ]
            }
        }
        }


lotteries_full = {
  "lottery_1": {
    "name": "Apple",
    "outcome_number": 4,
    "stake": "lo",
    "max_payoff": 44,
    "min_payoff": -46,
    "description": "n/a",
    "periods": {
      "0": [
        {
          "label": "Start",
          "probability": 1,
          "from": None,
          "abs_prob": 1
        }
      ],
      "1": [
        {
          "label": "+£2",
          "probability": 0.5,
          "from": "Start",
          "abs_prob": 0.5
        },
        {
          "label": "+£26",
          "probability": 0.5,
          "from": "Start",
          "abs_prob": 0.5
        }
      ],
      "2": [
        {
          "label": "+£6",
          "probability": 1,
          "from": "+£2",
          "abs_prob": 0.5
        },
        {
          "label": "-£29",
          "probability": 1,
          "from": "+£26",
          "abs_prob": 0.5
        }
      ],
      "3": [
        {
          "label": "+£36",
          "probability": 0.4,
          "from": "+£6",
          "parent": "+£2",
          "abs_prob": 0.2
        },
        {
          "label": "-£30",
          "probability": 0.6,
          "from": "+£6",
          "parent": "+£2",
          "abs_prob": 0.3
        },
        {
          "label": "+£39",
          "probability": 0.5,
          "from": "-£29",
          "parent": "£26",
          "abs_prob": 0.25
        },
        {
          "label": "-£43",
          "probability": 0.5,
          "from": "-£29",
          "parent": "£26",
          "abs_prob": 0.25
        }
      ]
    }
  },
  "lottery_2": {
  "name": "Banana",
  "outcome_number": 4,
  "stake": "lo",
  "max_payoff": 61,
  "min_payoff": -45,
  "description": "n/a",
  "periods": {
    "0": [
      {
        "label": "Start",
        "probability": 1,
        "from": None,
        "abs_prob": 1
      }
    ],
    "1": [
      {
        "label": "+£41",
        "probability": 0.6,
        "from": "Start",
        "abs_prob": 0.6
      },
      {
        "label": "+£22",
        "probability": 0.4,
        "from": "Start",
        "abs_prob": 0.4
      }
    ],
    "2": [
      {
        "label": "-£19",
        "probability": 1,
        "from": "+£41",
        "abs_prob": 0.6
      },
      {
        "label": "-£34",
        "probability": 1,
        "from": "+£22",
        "abs_prob": 0.4
      }
    ],
    "3": [
      {
        "label": "-£47",
        "probability": 0.7,
        "from": "-£19",
        "parent": "+£41",
        "abs_prob": 0.42
      },
      {
        "label": "+£39",
        "probability": 0.3,
        "from": "-£19",
        "parent": "+£41",
        "abs_prob": 0.18
      },
      {
        "label": "-£33",
        "probability": 0.4,
        "from": "-£34",
        "parent": "+£22",
        "abs_prob": 0.16
      },
      {
        "label": "+£36",
        "probability": 0.6,
        "from": "-£34",
        "parent": "+£22",
        "abs_prob": 0.24
      }
    ]
  }
},

"lottery_3": {
  "name": "Orange",
  "outcome_number": 4,
  "stake": "lo",
  "max_payoff": 49,
  "min_payoff": -47,
  "description": "n/a",
  "periods": {
    "0": [
      {
        "label": "Start",
        "probability": 1,
        "from": None,
        "abs_prob": 1
      }
    ],
    "1": [
      {
        "label": "+£8",
        "probability": 0.7,
        "from": "Start",
        "abs_prob": 0.7
      },
      {
        "label": "+£6",
        "probability": 0.3,
        "from": "Start",
        "abs_prob": 0.3
      }
    ],
    "2": [
      {
        "label": "+£10",
        "probability": 1,
        "from": "+£8",
        "abs_prob": 0.7
      },
      {
        "label": "-£17",
        "probability": 1,
        "from": "+£6",
        "abs_prob": 0.3
      }
    ],
    "3": [
      {
        "label": "-£37",
        "probability": 0.7,
        "from": "+£10",
        "parent": "+£8",
        "abs_prob": 0.49
      },
      {
        "label": "+£31",
        "probability": 0.3,
        "from": "+£10",
        "parent": "+£8",
        "abs_prob": 0.21
      },
      {
        "label": "+£32",
        "probability": 0.6,
        "from": "-£17",
        "parent": "+£6",
        "abs_prob": 0.18
      },
      {
        "label": "-£36",
        "probability": 0.4,
        "from": "-£17",
        "parent": "+£6",
        "abs_prob": 0.12
      }
    ]
  }
},
"lottery_4": {
  "name": "Pineapple",
  "outcome_number": 4,
  "stake": "hi",
  "max_payoff": 495,
  "min_payoff": -685,
  "description": "n/a",
  "periods": {
    "0": [
      {
        "label": "Start",
        "probability": 1,
        "from": None,
        "abs_prob": 1
      }
    ],
    "1": [
      {
        "label": "-£295",
        "probability": 0.1,
        "from": "Start",
        "abs_prob": 0.1
      },
      {
        "label": "+£110",
        "probability": 0.9,
        "from": "Start",
        "abs_prob": 0.9
      }
    ],
    "2": [
      {
        "label": "+£300",
        "probability": 1,
        "from": "-£295",
        "abs_prob": 0.1
      },
      {
        "label": "-£145",
        "probability": 1,
        "from": "+£110",
        "abs_prob": 0.9
      }
    ],
    "3": [
      {
        "label": "-£690",
        "probability": 0.6,
        "from": "+£300",
        "parent": "-£295",
        "abs_prob": 0.06
      },
      {
        "label": "+£260",
        "probability": 0.4,
        "from": "+£300",
        "parent": "-£295",
        "abs_prob": 0.04
      },
      {
        "label": "-£625",
        "probability": 0.4,
        "from": "-£145",
        "parent": "+£110",
        "abs_prob": 0.36
      },
      {
        "label": "+£530",
        "probability": 0.6,
        "from": "-£145",
        "parent": "+£110",
        "abs_prob": 0.54
      }
    ]
  }
},
"lottery_5": {
  "name": "Melon",
  "outcome_number": 4,
  "stake": "hi",
  "max_payoff": 395,
  "min_payoff": -715,
  "description": "n/a",
  "periods": {
    "0": [
      {
        "label": "Start",
        "probability": 1,
        "from": None,
        "abs_prob": 1
      }
    ],
    "1": [
      {
        "label": "-£260",
        "probability": 0.1,
        "from": "Start",
        "abs_prob": 0.1
      },
      {
        "label": "+£115",
        "probability": 0.9,
        "from": "Start",
        "abs_prob": 0.9
      }
    ],
    "2": [
      {
        "label": "+£110",
        "probability": 1,
        "from": "-£260",
        "abs_prob": 0.1
      },
      {
        "label": "-£145",
        "probability": 1,
        "from": "+£115",
        "abs_prob": 0.9
      }
    ],
    "3": [
      {
        "label": "+£400",
        "probability": 0.5,
        "from": "+£110",
        "parent": "-£260",
        "abs_prob": 0.05
      },
      {
        "label": "-£565",
        "probability": 0.5,
        "from": "+£110",
        "parent": "-£260",
        "abs_prob": 0.05
      },
      {
        "label": "+£425",
        "probability": 0.6,
        "from": "-£145",
        "parent": "+£115",
        "abs_prob": 0.54
      },
      {
        "label": "-£500",
        "probability": 0.4,
        "from": "-£145",
        "parent": "+£115",
        "abs_prob": 0.36
      }
    ]
  }
},
"lottery_6": {
  "name": "Coconut",
  "outcome_number": 4,
  "stake": "hi",
  "max_payoff": 335,
  "min_payoff": -725,
  "description": "n/a",
  "periods": {
    "0": [
      {
        "label": "Start",
        "probability": 1,
        "from": None,
        "abs_prob": 1
      }
    ],
    "1": [
      {
        "label": "+£120",
        "probability": 0.8,
        "from": "Start",
        "abs_prob": 0.8
      },
      {
        "label": "-£250",
        "probability": 0.2,
        "from": "Start",
        "abs_prob": 0.2
      }
    ],
    "2": [
      {
        "label": "-£115",
        "probability": 1,
        "from": "+£120",
        "abs_prob": 0.8
      },
      {
        "label": "+£120",
        "probability": 1,
        "from": "-£250",
        "abs_prob": 0.2
      }
    ],
    "3": [
      {
        "label": "+£210",
        "probability": 0.8,
        "from": "-£115",
        "parent": "+£120",
        "abs_prob": 0.64
      },
      {
        "label": "-£625",
        "probability": 0.2,
        "from": "-£115",
        "parent": "+£120",
        "abs_prob": 0.16
      },
      {
        "label": "-£595",
        "probability": 0.5,
        "from": "+£120",
        "parent": "-£250",
        "abs_prob": 0.10
      },
      {
        "label": "+£465",
        "probability": 0.5,
        "from": "+£120",
        "parent": "-£250",
        "abs_prob": 0.10
      }
    ]
  }
},
"lottery_7": {
  "name": "Dragonfruit",
  "outcome_number": 4,
  "stake": "hi",
  "max_payoff": 905,
  "min_payoff": -930,
  "description": "n/a",
  "periods": {
    "0": [
      {
        "label": "Start",
        "probability": 1,
        "from": None,
        "abs_prob": 1
      }
    ],
    "1": [
      {
        "label": "+£105",
        "probability": 0.9,
        "from": "Start",
        "abs_prob": 0.9
      },
      {
        "label": "+£245",
        "probability": 0.1,
        "from": "Start",
        "abs_prob": 0.1
      }
    ],
    "2": [
      {
        "label": "-£210",
        "probability": 1,
        "from": "+£105",
        "abs_prob": 0.9
      },
      {
        "label": "-£205",
        "probability": 1,
        "from": "+£245",
        "abs_prob": 0.1
      }
    ],
    "3": [
      {
        "label": "-£825",
        "probability": 0.3,
        "from": "-£210",
        "parent": "+£105",
        "abs_prob": 0.27
      },
      {
        "label": "+£495",
        "probability": 0.7,
        "from": "-£210",
        "parent": "+£105",
        "abs_prob": 0.63
      },
      {
        "label": "-£565",
        "probability": 0.6,
        "from": "-£205",
        "parent": "+£245",
        "abs_prob": 0.06
      },
      {
        "label": "+£865",
        "probability": 0.4,
        "from": "-£205",
        "parent": "+£245",
        "abs_prob": 0.04
      }
    ]
  }
},
"lottery_8": {
  "name": "Plum",
  "outcome_number": 4,
  "stake": "hi",
  "max_payoff": 755,
  "min_payoff": -1315,
  "description": "n/a",
  "periods": {
    "0": [
      {
        "label": "Start",
        "probability": 1,
        "from": None,
        "abs_prob": 1
      }
    ],
    "1": [
      {
        "label": "+£200",
        "probability": 0.9,
        "from": "Start",
        "abs_prob": 0.9
      },
      {
        "label": "+£125",
        "probability": 0.1,
        "from": "Start",
        "abs_prob": 0.1
      }
    ],
    "2": [
      {
        "label": "-£365",
        "probability": 1,
        "from": "+£200",
        "abs_prob": 0.9
      },
      {
        "label": "-£515",
        "probability": 1,
        "from": "+£125",
        "abs_prob": 0.1
      }
    ],
    "3": [
      {
        "label": "+£920",
        "probability": 0.6,
        "from": "-£365",
        "parent": "+£200",
        "abs_prob": 0.54
      },
      {
        "label": "-£985",
        "probability": 0.4,
        "from": "-£365",
        "parent": "+£200",
        "abs_prob": 0.36
      },
      {
        "label": "-£925",
        "probability": 0.5,
        "from": "-£515",
        "parent": "+£125",
        "abs_prob": 0.05
      },
      {
        "label": "-£670",
        "probability": 0.5,
        "from": "-£515",
        "parent": "+£125",
        "abs_prob": 0.05
      }
    ]
  }
},
"lottery_9": {
  "name": "Durian",
  "outcome_number": 4,
  "stake": "hi",
  "max_payoff": 821,
  "min_payoff": -560,
  "description": "n/a",
  "periods": {
    "0": [
      {
        "label": "Start",
        "probability": 1,
        "from": None,
        "abs_prob": 1
      }
    ],
    "1": [
      {
        "label": "+£370",
        "probability": 0.2,
        "from": "Start",
        "abs_prob": 0.2
      },
      {
        "label": "-£125",
        "probability": 0.8,
        "from": "Start",
        "abs_prob": 0.8
      }
    ],
    "2": [
      {
        "label": "-£190",
        "probability": 1,
        "from": "+£370",
        "abs_prob": 0.2
      },
      {
        "label": "+£165",
        "probability": 1,
        "from": "-£125",
        "abs_prob": 0.8
      }
    ],
    "3": [
      {
        "label": "+£641",
        "probability": 0.6,
        "from": "-£190",
        "parent": "+£370",
        "abs_prob": 0.12
      },
      {
        "label": "-£735",
        "probability": 0.4,
        "from": "-£190",
        "parent": "+£370",
        "abs_prob": 0.08
      },
      {
        "label": "+£715",
        "probability": 0.4,
        "from": "+£165",
        "parent": "-£125",
        "abs_prob": 0.32
      },
      {
        "label": "-£600",
        "probability": 0.6,
        "from": "+£165",
        "parent": "-£125",
        "abs_prob": 0.48
      }
    ]
  }
},

"lottery_10": {
  "name": "Pomegranate",
  "outcome_number": 6,
  "stake": "lo",
  "max_payoff": 45,
  "min_payoff": -42,
  "description": "n/a",
  "periods": {
    "0": [
      {
        "label": "Start",
        "probability": 1,
        "from": None,
        "abs_prob": 1
      }
    ],
    "1": [
      {
        "label": "-£1",
        "probability": 0.3,
        "from": "Start",
        "abs_prob": 0.3
      },
      {
        "label": "+£11",
        "probability": 0.7,
        "from": "Start",
        "abs_prob": 0.7
      }
    ],
    "2": [
      {
        "label": "-£1",
        "probability": 1,
        "from": "-£1",
        "abs_prob": 0.3
      },
      {
        "label": "-£7",
        "probability": 0.4,
        "from": "+£11",
        "abs_prob": 0.28
      },
      {
        "label": "-£2",
        "probability": 0.6,
        "from": "+£11",
        "abs_prob": 0.42
      }
    ],
    "3": [
      {
        "label": "-£40",
        "probability": 0.6,
        "from": "-£1",
        "parent": "-£1",
        "abs_prob": 0.18
      },
      {
        "label": "+£47",
        "probability": 0.4,
        "from": "-£1",
        "parent": "-£1",
        "abs_prob": 0.12
      },
      {
        "label": "+£27",
        "probability": 0.3,
        "from": "-£7",
        "parent": "+£11",
        "abs_prob": 0.084
      },
      {
        "label": "-£20",
        "probability": 0.7,
        "from": "-£7",
        "parent": "+£11",
        "abs_prob": 0.196
      },
      {
        "label": "-£26",
        "probability": 0.2,
        "from": "-£2",
        "parent": "+£11",
        "abs_prob": 0.084
      },
      {
        "label": "+£3",
        "probability": 0.8,
        "from": "-£2",
        "parent": "+£11",
        "abs_prob": 0.336
      }
    ]
  }
},
"lottery_11": {
  "name": "Avocado",
  "outcome_number": 6,
  "stake": "lo",
  "max_payoff": 42,
  "min_payoff": -34,
  "description": "n/a",
  "periods": {
    "0": [
      {
        "label": "Start",
        "probability": 1,
        "from": None,
        "abs_prob": 1
      }
    ],
    "1": [
      {
        "label": "+£4",
        "probability": 0.1,
        "from": "Start",
        "abs_prob": 0.1
      },
      {
        "label": "+£5",
        "probability": 0.9,
        "from": "Start",
        "abs_prob": 0.9
      }
    ],
    "2": [
      {
        "label": "-£2",
        "probability": 1,
        "from": "+£4",
        "abs_prob": 0.1
      },
      {
        "label": "+£6",
        "probability": 0.7,
        "from": "+£5",
        "abs_prob": 0.63
      },
      {
        "label": "-£6",
        "probability": 0.3,
        "from": "+£5",
        "abs_prob": 0.27
      }
    ],
    "3": [
      {
        "label": "+£32",
        "probability": 0.3,
        "from": "-£2",
        "parent": "+£4",
        "abs_prob": 0.03
      },
      {
        "label": "-£36",
        "probability": 0.7,
        "from": "-£2",
        "parent": "+£4",
        "abs_prob": 0.07
      },
      {
        "label": "-£38",
        "probability": 0.6,
        "from": "+£6",
        "parent": "+£5",
        "abs_prob": 0.378
      },
      {
        "label": "+£31",
        "probability": 0.4,
        "from": "+£6",
        "parent": "+£5",
        "abs_prob": 0.252
      },
      {
        "label": "+£16",
        "probability": 0.7,
        "from": "-£6",
        "parent": "+£5",
        "abs_prob": 0.189
      },
      {
        "label": "-£27",
        "probability": 0.3,
        "from": "-£6",
        "parent": "+£5",
        "abs_prob": 0.081
      }
    ]
  }
},

"lottery_12": {
  "name": "Raspberry",
  "outcome_number": 6,
  "stake": "lo",
  "max_payoff": 36,
  "min_payoff": -49,
  "description": "n/a",
  "periods": {
    "0": [
      {
        "label": "Start",
        "probability": 1,
        "from": None,
        "abs_prob": 1
      }
    ],
    "1": [
      {
        "label": "-£3",
        "probability": 0.1,
        "from": "Start",
        "abs_prob": 0.1
      },
      {
        "label": "+£13",
        "probability": 0.9,
        "from": "Start",
        "abs_prob": 0.9
      }
    ],
    "2": [
      {
        "label": "-£13",
        "probability": 1,
        "from": "-£3",
        "abs_prob": 0.1
      },
      {
        "label": "+£7",
        "probability": 0.3,
        "from": "+£13",
        "abs_prob": 0.27
      },
      {
        "label": "+£2",
        "probability": 0.7,
        "from": "+£13",
        "abs_prob": 0.63
      }
    ],
    "3": [
      {
        "label": "-£33",
        "probability": 0.5,
        "from": "-£13",
        "parent": "-£3",
        "abs_prob": 0.05
      },
      {
        "label": "+£35",
        "probability": 0.5,
        "from": "-£13",
        "parent": "-£3",
        "abs_prob": 0.05
      },
      {
        "label": "-£43 ",
        "probability": 0.6,
        "from": "+£7",
        "parent": "+£13",
        "abs_prob": 0.162
      },
      {
        "label": "+£1",
        "probability": 0.4,
        "from": "+£7",
        "parent": "+£13",
        "abs_prob": 0.108
      },
      {
        "label": "-£43",
        "probability": 0.5,
        "from": "+£2",
        "parent": "+£13",
        "abs_prob": 0.315
      },
      {
        "label": "+£21",
        "probability": 0.5,
        "from": "+£2",
        "parent": "+£13",
        "abs_prob": 0.315
      }
    ]
  }
},
"lottery_13": {
  "name": "Fig",
  "outcome_number": 6,
  "stake": "lo",
  "max_payoff": 47,
  "min_payoff": -40,
  "description": "n/a",
  "periods": {
    "0": [
      {
        "label": "Start",
        "probability": 1,
        "from": None,
        "abs_prob": 1
      }
    ],
    "1": [
      {
        "label": "-£1",
        "probability": 0.2,
        "from": "Start",
        "abs_prob": 0.2
      },
      {
        "label": "+£5",
        "probability": 0.8,
        "from": "Start",
        "abs_prob": 0.8
      }
    ],
    "2": [
      {
        "label": "+£5",
        "probability": 1,
        "from": "-£1",
        "abs_prob": 0.2
      },
      {
        "label": "+£4",
        "probability": 0.2,
        "from": "+£5",
        "abs_prob": 0.16
      },
      {
        "label": "+£0",
        "probability": 0.8,
        "from": "+£5",
        "abs_prob": 0.64
      }
    ],
    "3": [
      {
        "label": "-£32",
        "probability": 0.5,
        "from": "+£5",
        "parent": "-£1",
        "abs_prob": 0.10
      },
      {
        "label": "+£33",
        "probability": 0.5,
        "from": "+£5",
        "parent": "-£1",
        "abs_prob": 0.10
      },
      {
        "label": "+£7",
        "probability": 0.3,
        "from": "+£4",
        "parent": "+£5",
        "abs_prob": 0.048
      },
      {
        "label": "-£47",
        "probability": 0.7,
        "from": "+£4",
        "parent": "+£5",
        "abs_prob": 0.112
      },
      {
        "label": "-£45",
        "probability": 0.5,
        "from": "+£0",
        "parent": "+£5",
        "abs_prob": 0.32
      },
      {
        "label": "+£42",
        "probability": 0.5,
        "from": "+£0",
        "parent": "+£5",
        "abs_prob": 0.32
      }
    ]
  }
},

"lottery_14": {
  "name": "Papaya",
  "outcome_number": 6,
  "stake": "lo",
  "max_payoff": 44,
  "min_payoff": -39,
  "description": "n/a",
  "periods": {
    "0": [
      {
        "label": "Start",
        "probability": 1,
        "from": None,
        "abs_prob": 1
      }
    ],
    "1": [
      {
        "label": "-£9",
        "probability": 0.2,
        "from": "Start",
        "abs_prob": 0.2
      },
      {
        "label": "+£9",
        "probability": 0.8,
        "from": "Start",
        "abs_prob": 0.8
      }
    ],
    "2": [
      {
        "label": "-£8",
        "probability": 1,
        "from": "-£9",
        "abs_prob": 0.2
      },
      {
        "label": "-£17",
        "probability": 0.2,
        "from": "+£9",
        "abs_prob": 0.16
      },
      {
        "label": "-£5",
        "probability": 0.8,
        "from": "+£9",
        "abs_prob": 0.64
      }
    ],
    "3": [
      {
        "label": "+£47",
        "probability": 0.3,
        "from": "-£8",
        "parent": "-£9",
        "abs_prob": 0.06
      },
      {
        "label": "-£22 ",
        "probability": 0.7,
        "from": "-£8",
        "parent": "-£9",
        "abs_prob": 0.14
      },
      {
        "label": "-£31",
        "probability": 0.5,
        "from": "-£17",
        "parent": "+£9",
        "abs_prob": 0.08
      },
      {
        "label": "+£40",
        "probability": 0.5,
        "from": "-£17",
        "parent": "+£9",
        "abs_prob": 0.08
      },
      {
        "label": "-£22",
        "probability": 0.6,
        "from": "-£5",
        "parent": "+£9",
        "abs_prob": 0.384
      },
      {
        "label": "+£40",
        "probability": 0.4,
        "from": "-£5",
        "parent": "+£9",
        "abs_prob": 0.256
      }
    ]
  }
}

}
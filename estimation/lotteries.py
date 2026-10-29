"""Lottery trees shown in the experiment (version 2, 16 lotteries).

Each lottery lists its nodes per period; node labels are payoffs such as
"+£26", ``from``/``parent`` name the ancestors, ``abs_prob`` is the path
probability. ``lottery_paths.transform`` turns a tree into payoff streams.
"""

LOTTERIES_V2 = {


  "cali_gain": {
    "name": "cali_gain",
    "outcome_number": 2,
    "stake": "lo",
    "max_payoff": 40,
    "min_payoff": 12,
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
          "label": "+£40",
          "probability": 0.9,
          "from": "Start",
          "abs_prob": 0.9
        },
        {
          "label": "+£12",
          "probability": 0.1,
          "from": "Start",
          "abs_prob": 0.1
        }
      ]
    }
  },
  "cali_gain_st": {
    "name": "cali_gain_st",
    "outcome_number": 2,
    "stake": "lo",
    "max_payoff": 40,
    "min_payoff": 12,
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
          "label": "+£40",
          "probability": 0.5,
          "from": "Start",
          "abs_prob": 0.5
        },
        {
          "label": "+£12",
          "probability": 0.5,
          "from": "Start",
          "abs_prob": 0.5
        }
      ]
    }
  },

"cali_gain_de": {
  "name": "cali_gain_de",
  "outcome_number": 4,
  "stake": "lo",
  "max_payoff": 40,
  "min_payoff": 12,
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
        "label": "+£10",
        "probability": 0.9,
        "from": "Start",
        "abs_prob": 0.9
      },
      {
        "label": "+£3",
        "probability": 0.1,
        "from": "Start",
        "abs_prob": 0.1
      }
    ],
    "2": [
      {
        "label": "+£30",
        "probability": 1,
        "from": "+£10",
        "abs_prob": 0.9
      },
      {
        "label": "+£9",
        "probability": 1,
        "from": "+£3",
        "abs_prob": 0.1
      }
    ]
  }
},

  "cali_loss": {
    "name": "cali_loss",
    "outcome_number": 2,
    "stake": "lo",
    "max_payoff": -8,
    "min_payoff": -60,
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
          "label": "-£8",
          "probability": 0.6,
          "from": "Start",
          "abs_prob": 0.6
        },
        {
          "label": "-£60",
          "probability": 0.4,
          "from": "Start",
          "abs_prob": 0.4
        }
      ]
    }
  },
  "cali_loss_st": {
    "name": "cali_loss_st",
    "outcome_number": 2,
    "stake": "lo",
    "max_payoff": -8,
    "min_payoff": -60,
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
          "label": "-£8",
          "probability": 0.8,
          "from": "Start",
          "abs_prob": 0.8
        },
        {
          "label": "-£60",
          "probability": 0.2,
          "from": "Start",
          "abs_prob": 0.2
        }
      ]
    }
  },

"cali_loss_de": {
  "name": "cali_loss_de",
  "outcome_number": 4,
  "stake": "lo",
  "max_payoff": -8,
  "min_payoff": -60,
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
        "label": "-£2",
        "probability": 0.6,
        "from": "Start",
        "abs_prob": 0.6
      },
      {
        "label": "-£15",
        "probability": 0.4,
        "from": "Start",
        "abs_prob": 0.4
      }
    ],
    "2": [
      {
        "label": "-£6",
        "probability": 1,
        "from": "-£2",
        "abs_prob": 0.6
      },
      {
        "label": "-£45",
        "probability": 1,
        "from": "-£15",
        "abs_prob": 0.4
      }
    ]
  }
},
  "cali_mix": {
    "name": "cali_mix",
    "outcome_number": 2,
    "stake": "hi",
    "max_payoff": 700,
    "min_payoff": -100,
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
          "label": "+£700",
          "probability": 0.5,
          "from": "Start",
          "abs_prob": 0.5
        },
        {
          "label": "-£100",
          "probability": 0.5,
          "from": "Start",
          "abs_prob": 0.5
        }
      ]
    }
  },
  "cali_mix_pres": {
    "name": "cali_mix_pres",
    "outcome_number": 3,
    "stake": "hi",
    "max_payoff": 700,
    "min_payoff": -100,
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
          "label": "+£700",
          "probability": 0.4,
          "from": "Start",
          "abs_prob": 0.4
        },
        {
          "label": "+£300",
          "probability": 0.2,
          "from": "Start",
          "abs_prob": 0.2
        },
        {
          "label": "-£100",
          "probability": 0.4,
          "from": "Start",
          "abs_prob": 0.4
        }
      ]
    }
  },

"lottery_1": {
    "name": "lottery_1",
    "outcome_number": 4,
    "stake": "lo",
    "max_payoff": 19,
    "min_payoff": 11,
    "description": "treatment",
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
          "label": "+£11",
          "probability": 0.5,
          "from": "Start",
          "abs_prob": 0.5
        },
        {
          "label": "+£8",
          "probability": 0.5,
          "from": "Start",
          "abs_prob": 0.5
        }
      ],
      "2": [
        {
          "label": "+£2",
          "probability": 1,
          "from": "+£11",
          "abs_prob": 0.4
        },
        {
          "label": "+£1",
          "probability": 1,
          "from": "+£8",
          "abs_prob": 0.6
        }
      ],
      "3": [
        {
          "label": "+£6",
          "probability": 0.9,
          "from": "+£2",
          "parent": "+£11",
          "abs_prob": 0.45
        },
        {
          "label": "+£5",
          "probability": 0.1,
          "from": "+£2",
          "parent": "+£11",
          "abs_prob": 0.05
        },
        {
          "label": "+£2",
          "probability": 0.4,
          "from": "+£1",
          "parent": "+£8",
          "abs_prob": 0.2
        },
        {
          "label": "+£6",
          "probability": 0.6,
          "from": "+£1",
          "parent": "+£8",
          "abs_prob": 0.3
        }
      ]
    }
  },

  "lottery_2": {
    "name": "Lottery 2",
    "outcome_number": 4,
    "stake": "hi",
    "max_payoff": 2155,
    "min_payoff": 1640,
    "description": "gains",
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
          "label": "+£675",
          "probability": 0.4,
          "from": "Start",
          "abs_prob": 0.4
        },
        {
          "label": "+£520",
          "probability": 0.6,
          "from": "Start",
          "abs_prob": 0.6
        }
      ],
      "2": [
        {
          "label": "+£140",
          "probability": 1,
          "from": "+£675",
          "abs_prob": 0.4
        },
        {
          "label": "+£920",
          "probability": 1,
          "from": "+£520",
          "abs_prob": 0.6
        }
      ],
      "3": [
        {
          "label": "+£845",
          "probability": 0.1,
          "from": "+£140",
          "parent": "+£675",
          "abs_prob": 0.04
        },
        {
          "label": "+£825",
          "probability": 0.9,
          "from": "+£140",
          "parent": "+£675",
          "abs_prob": 0.36
        },
        {
          "label": "+£715",
          "probability": 0.6,
          "from": "+£920",
          "parent": "+£520",
          "abs_prob": 0.36
        },
        {
          "label": "+£640",
          "probability": 0.4,
          "from": "+£920",
          "parent": "+£520",
          "abs_prob": 0.24
        }
      ]
    }
  },

  "lottery_3": {
    "name": "Lottery 3",
    "outcome_number": 4,
    "stake": "lo",
    "max_payoff": 22,
    "min_payoff": 9,
    "description": "treatment",
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
          "probability": 0.4,
          "from": "Start",
          "abs_prob": 0.4
        },
        {
          "label": "+£10",
          "probability": 0.6,
          "from": "Start",
          "abs_prob": 0.6
        }
      ],
      "2": [
        {
          "label": "+£3",
          "probability": 1,
          "from": "+£4",
          "abs_prob": 0.4
        },
        {
          "label": "+£3",
          "probability": 1,
          "from": "+£10",
          "abs_prob": 0.6
        }
      ],
      "3": [
        {
          "label": "+£9",
          "probability": 0.6,
          "from": "+£3",
          "parent": "+£4",
          "abs_prob": 0.24
        },
        {
          "label": "+£2",
          "probability": 0.4,
          "from": "+£3",
          "parent": "+£4",
          "abs_prob": 0.16
        },
        {
          "label": "+£9",
          "probability": 0.5,
          "from": "+£3",
          "parent": "+£10",
          "abs_prob": 0.3
        },
        {
          "label": "+£7",
          "probability": 0.5,
          "from": "+£3",
          "parent": "+£10",
          "abs_prob": 0.3
        }
      ]
    }
  },

  "lottery_4": {
    "name": "Lottery 4",
    "outcome_number": 4,
    "stake": "lo",
    "max_payoff": -24,
    "min_payoff": -90,
    "description": "losses",
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
          "label": "-£50",
          "probability": 0.6,
          "from": "Start",
          "abs_prob": 0.6
        },
        {
          "label": "-£39",
          "probability": 0.4,
          "from": "Start",
          "abs_prob": 0.4
        }
      ],
      "2": [
        {
          "label": "-£17",
          "probability": 1,
          "from": "-£50",
          "abs_prob": 0.6
        },
        {
          "label": "+£47",
          "probability": 1,
          "from": "-£39",
          "abs_prob": 0.4
        }
      ],
      "3": [
        {
          "label": "-£1",
          "probability": 0.8,
          "from": "-£17",
          "parent": "-£50",
          "abs_prob": 0.48
        },
        {
          "label": "-£23",
          "probability": 0.2,
          "from": "-£17",
          "parent": "-£50",
          "abs_prob": 0.12
        },
        {
          "label": "-£45",
          "probability": 0.2,
          "from": "+£47",
          "parent": "-£39",
          "abs_prob": 0.08
        },
        {
          "label": "-£32",
          "probability": 0.8,
          "from": "+£47",
          "parent": "-£39",
          "abs_prob": 0.32
        }
      ]
    }
  },

  "lottery_5": {
    "name": "Lottery 5",
    "outcome_number": 4,
    "stake": "lo",
    "max_payoff": -63,
    "min_payoff": -114,
    "description": "losses",
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
          "label": "-£44",
          "probability": 0.5,
          "from": "Start",
          "abs_prob": 0.5
        },
        {
          "label": "-£30",
          "probability": 0.5,
          "from": "Start",
          "abs_prob": 0.5
        }
      ],
      "2": [
        {
          "label": "-£35",
          "probability": 1,
          "from": "-£44",
          "abs_prob": 0.5
        },
        {
          "label": "-£29",
          "probability": 1,
          "from": "-£30",
          "abs_prob": 0.5
        }
      ],
      "3": [
        {
          "label": "-£33",
          "probability": 0.5,
          "from": "-£35",
          "parent": "-£44",
          "abs_prob": 0.25
        },
        {
          "label": "-£35",
          "probability": 0.5,
          "from": "-£35",
          "parent": "-£44",
          "abs_prob": 0.25
        },
        {
          "label": "-£4",
          "probability": 0.5,
          "from": "-£29",
          "parent": "-£30",
          "abs_prob": 0.25
        },
        {
          "label": "-£38",
          "probability": 0.5,
          "from": "-£29",
          "parent": "-£30",
          "abs_prob": 0.25
        }
      ]
    }
  },

  "lottery_6": {
    "name": "Lottery 6",
    "outcome_number": 4,
    "stake": "lo",
    "max_payoff": 62,
    "min_payoff": -73,
    "description": "mixed",
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
          "label": "-£19",
          "probability": 0.7,
          "from": "Start",
          "abs_prob": 0.7
        },
        {
          "label": "-£43",
          "probability": 0.3,
          "from": "Start",
          "abs_prob": 0.3
        }
      ],
      "2": [
        {
          "label": "+£50",
          "probability": 1,
          "from": "-£19",
          "abs_prob": 0.7
        },
        {
          "label": "-£40",
          "probability": 1,
          "from": "-£43",
          "abs_prob": 0.3
        }
      ],
      "3": [
        {
          "label": "+£31",
          "probability": 0.8,
          "from": "+£50",
          "parent": "-£19",
          "abs_prob": 0.56
        },
        {
          "label": "-£11",
          "probability": 0.2,
          "from": "+£50",
          "parent": "-£19",
          "abs_prob": 0.14
        },
        {
          "label": "+£10",
          "probability": 0.5,
          "from": "-£40",
          "parent": "-£43",
          "abs_prob": 0.15
        },
        {
          "label": "+£14",
          "probability": 0.5,
          "from": "-£40",
          "parent": "-£43",
          "abs_prob": 0.15
        }
      ]
    }
  },

  "lottery_7": {
    "name": "Lottery 7",
    "outcome_number": 4,
    "stake": "hi",
    "max_payoff": -1410,
    "min_payoff": -2010,
    "description": "losses",
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
          "label": "-£315",
          "probability": 0.6,
          "from": "Start",
          "abs_prob": 0.6
        },
        {
          "label": "-£795",
          "probability": 0.4,
          "from": "Start",
          "abs_prob": 0.4
        }
      ],
      "2": [
        {
          "label": "-£695",
          "probability": 1,
          "from": "-£315",
          "abs_prob": 0.6
        },
        {
          "label": "-£345",
          "probability": 1,
          "from": "-£795",
          "abs_prob": 0.4
        }
      ],
      "3": [
        {
          "label": "-£1000",
          "probability": 0.2,
          "from": "-£695",
          "parent": "-£315",
          "abs_prob": 0.12
        },
        {
          "label": "-£400",
          "probability": 0.8,
          "from": "-£695",
          "parent": "-£315",
          "abs_prob": 0.48
        },
        {
          "label": "-£485",
          "probability": 0.7,
          "from": "-£345",
          "parent": "-£795",
          "abs_prob": 0.28
        },
        {
          "label": "-£775",
          "probability": 0.3,
          "from": "-£345",
          "parent": "-£795",
          "abs_prob": 0.12
        }
      ]
    }
  },

  "lottery_8": {
    "name": "Lottery 8",
    "outcome_number": 4,
    "stake": "hi",
    "max_payoff": 1125,
    "min_payoff": -535,
    "description": "mixed",
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
          "label": "-£230",
          "probability": 0.3,
          "from": "Start",
          "abs_prob": 0.3
        },
        {
          "label": "+£425",
          "probability": 0.7,
          "from": "Start",
          "abs_prob": 0.7
        }
      ],
      "2": [
        {
          "label": "+£820",
          "probability": 1,
          "from": "-£230",
          "abs_prob": 0.3
        },
        {
          "label": "-£100",
          "probability": 1,
          "from": "+£425",
          "abs_prob": 0.7
        }
      ],
      "3": [
        {
          "label": "+£435",
          "probability": 0.4,
          "from": "+£820",
          "parent": "-£230",
          "abs_prob": 0.12
        },
        {
          "label": "+£535",
          "probability": 0.6,
          "from": "+£820",
          "parent": "-£230",
          "abs_prob": 0.18
        },
        {
          "label": "+£160",
          "probability": 0.2,
          "from": "-£100",
          "parent": "+£425",
          "abs_prob": 0.14
        },
        {
          "label": "-£860",
          "probability": 0.8,
          "from": "-£100",
          "parent": "+£425",
          "abs_prob": 0.56
        }
      ]
    }
  }
}

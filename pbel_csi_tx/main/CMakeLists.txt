idf_component_register(
    SRCS "tx_main.c"
    INCLUDE_DIRS "."
    REQUIRES esp_wifi esp_event esp_netif nvs_flash esp_hw_support
)
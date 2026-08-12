def allegro_kuka_small_cuboid_scales(base_size: float = 0.05):
    """Return deterministic metric scales from Allegro-Kuka's small pool."""
    scale_percentages = [100, 50, 66, 75, 90, 110, 125, 150, 175, 200, 250, 300]
    scales = []
    for x_scale in scale_percentages:
        for y_scale in scale_percentages:
            for z_scale in scale_percentages:
                normalized_volume = x_scale * y_scale * z_scale / 1_000_000
                if 1.0 <= normalized_volume <= 2.5:
                    scales.append(
                        (
                            base_size * x_scale / 100,
                            base_size * y_scale / 100,
                            base_size * z_scale / 100,
                        )
                    )
    return scales

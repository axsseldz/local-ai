"use client";

import { useEffect, useState } from "react";

function uniqueId(): number {
    return Math.floor(Math.random() * 1_000_000_000);
}

type Blip = {
    id: number;
    x: number;
    y: number;
};

export default function RadarBackground() {
    const [blips, setBlips] = useState<Blip[]>([]);

    useEffect(() => {
        const interval = window.setInterval(() => {
            const id = uniqueId();
            const angle = Math.random() * Math.PI * 2;
            const radius = 10 + Math.random() * 30;
            const x = 50 + radius * Math.cos(angle);
            const y = 40 + radius * Math.sin(angle);

            setBlips((prev) => {
                const trimmed = prev.length > 7 ? prev.slice(prev.length - 7) : prev;
                return [...trimmed, { id, x, y }];
            });

            window.setTimeout(() => {
                setBlips((prev) => prev.filter((b) => b.id !== id));
            }, 1600);
        }, 2200);

        return () => window.clearInterval(interval);
    }, []);

    return (
        <div className="radar-container">
            <div className="radar-ping"></div>
            <div className="radar-ping"></div>
            <div className="radar-ping"></div>
            <div className="radar-ping radar-ping-strong"></div>
            {blips.map((b) => (
                <div
                    key={b.id}
                    className="radar-blip"
                    style={{ left: `${b.x}%`, top: `${b.y}%` }}
                />
            ))}
        </div>
    );
}

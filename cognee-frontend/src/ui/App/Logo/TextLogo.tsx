export default function TextLogo({ width = 158, height = 44, color = 'currentColor', className = '' }) {
  return (
    <svg
      width={width}
      height={height}
      viewBox="0 0 158 44"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      <text
        x="50%"
        y="50%"
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="32"
        fill={color}
        fontFamily='"Dancing Script", cursive'
      >
        国传智能
      </text>
    </svg>
  );
}

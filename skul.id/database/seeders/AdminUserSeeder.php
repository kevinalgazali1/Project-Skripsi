<?php

namespace Database\Seeders;

use Illuminate\Database\Seeder;
use App\Models\User;
use Illuminate\Support\Facades\Hash;

class AdminUserSeeder extends Seeder
{
    public function run(): void
    {
        User::updateOrCreate(
            ['no_hp' => '082212345678'],
            [
                'password' => Hash::make('admin123'),
                'role' => 'admin',
            ]
        );
    }
}
